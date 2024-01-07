import warnings
from datetime import datetime
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from kaggle_prediction_interval_birthweight.data.constants import (
    BIN_LABELS,
    MISSING_CODE,
    SOFTPLUS_SCALE,
    VARIABLE_TYPE,
    Y_MEAN,
    Y_SD,
)
from kaggle_prediction_interval_birthweight.utils.utils import np_softplus_inv

TIMESTAMP_COL_NAME = "DOB_TT"


class DataProcessor:
    """Class with methods for processing data."""

    def __init__(
        self,
        model_type: str,
        standardization_parameters: Optional[Dict[str, np.ndarray]] = None,
        feature_categories: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        model_type: str
            One of "RidgeRegressor", "HistBoostRegressor", "WildWoodRegressor",
            "MissingnessNeuralNetRegressor", "MissingnessNeuralNetEIM", or
            "MissingnessNeuralNetClassifier". Determines how data are imputed and standardized
            before modeling.
        standardization_parameters: Dict
            Precomputed columnwise means and sds or decorrelation matrix, if already computed.
            None during training, but at test time these values should be passed in.
        feature_categories: Dict
            levels of categorical variables. None during training (they are inferred), but at
            test time these values should be passed in.
        """
        allowed_model_types = [
            "RidgeRegressor",
            "HistBoostRegressor",
            "WildWoodRegressor",
            "MissingnessNeuralNetRegressor",
            "MissingnessNeuralNetClassifier",
            "MissingnessNeuralNetEIM",
        ]
        if model_type not in allowed_model_types:
            raise NotImplementedError(f"Supported model_type values: {allowed_model_types}")
        self.model_type = model_type
        self.standardization_parameters = standardization_parameters
        self.feature_categories = feature_categories
        self.nondegenerate_columns = None

    def _enforce_feature_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce feature types.

        Parameters
        ----------
        df: pd.DataFrame
            The input data frame

        Returns
        -------
        pd.DataFrame
            df with categorical features as strings and numerical features as floats.
        """
        df = df.copy()

        # convert that timestamp column to epoch. First, as a string
        fake_datetime = "1970-01-01 " + df[TIMESTAMP_COL_NAME].astype(str).str.zfill(4)

        # preserve the missing values, but create a new code for it
        fake_datetime = fake_datetime.apply(
            lambda x: None if str(MISSING_CODE[TIMESTAMP_COL_NAME]) in x else x
        )

        # now compute the time in seconds from the beginning of their birth year
        df[TIMESTAMP_COL_NAME] = (
            pd.to_datetime(fake_datetime, format="%Y-%m-%d %H%M") - datetime(1970, 1, 1)
        ).dt.total_seconds()

        # replace missing values with a large number, and save that for later
        max_seconds = df[TIMESTAMP_COL_NAME].max()
        missing_code = int("9" * (len(str(max_seconds + 1))))
        df[TIMESTAMP_COL_NAME] = df[TIMESTAMP_COL_NAME].fillna(missing_code)
        self.missing_timestamp_code = missing_code

        # convert implicit to explicit categorical columns
        for feature in VARIABLE_TYPE.keys():
            if "categorical" in VARIABLE_TYPE[feature]:
                df[feature] = df[feature].astype(str)
            else:
                df[feature] = df[feature].astype(float)
        return df

    def _process_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process missing data.

        Parameters
        ----------
        df: pd.DataFrame
            The input data frame

        Returns
        -------
        pd.DataFrame
            The same data frame with missing data processed according to self.model_type.
        """
        df = df.copy()
        for feature in MISSING_CODE.keys():
            # handle lists of multiple missing codes, where relevant
            missing_code = (
                MISSING_CODE[feature]
                if isinstance(MISSING_CODE[feature], list)
                else [MISSING_CODE[feature]]
            )

            # handle the timestamp feature that has a new missing code
            missing_code = (
                [self.missing_timestamp_code] if feature == TIMESTAMP_COL_NAME else missing_code
            )

            # for linear regression, impute with something reasonable-- medians and modes.
            if self.model_type == "RidgeRegressor":
                if "categorical" in VARIABLE_TYPE[feature]:
                    replacement = df.loc[~df[feature].isin(missing_code), feature].mode()
                else:
                    replacement = df.loc[~df[feature].isin(missing_code), feature].median()

            # for the neural network, use real NAs with the expected relu layer. Boosting also has
            # native support for NaNs.
            elif self.model_type in [
                "MissingnessNeuralNetRegressor",
                "MissingnessNeuralNetClassifier",
                "MissingnessNeuralNetEIM",
                "HistBoostRegressor",
                "WildWoodRegressor",
            ]:
                replacement = None
            df.loc[df[feature].isin(missing_code), feature] = replacement

        return df

    def _subset_and_binarize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop unwanted columns and binarize selected features.

        Parameters
        ----------
        df: pd.DataFrame
            The input data frame

        Returns
        -------
        pd.DataFrame
            The data frame with removed and binarized columns.
        """
        df = df.copy()
        # during the EDA, we decided to drop these
        keepers = list(set(VARIABLE_TYPE.keys()) - {"PAY", "ILP_R"})
        df = df.drop(["PAY", "ILP_R"], axis=1)

        # we also added a polynomial effect for one (transformed) feature
        df["ILOP_R"] = np.log(df["ILOP_R"] + 0.1)
        df["ILOP_R_2"] = df["ILOP_R"] ** 2
        df["ILOP_R_3"] = df["ILOP_R"] ** 3
        keepers = keepers + ["ILOP_R_2", "ILOP_R_3"]

        # make gestation period explicit
        gestation_guess = 9 if self.model_type == "RidgeRegressor" else np.nan
        df["gestation_time"] = np.where(
            df["DLMP_MM"] != MISSING_CODE["DLMP_MM"], df["DOB_MM"] - df["DLMP_MM"], gestation_guess
        )
        df["gestation_time"] = np.where(
            df["gestation_time"] < 0, df["gestation_time"] + 12, df["gestation_time"]
        )
        keepers = keepers + ["gestation_time"]

        numeric_features = list(
            set(keepers)
            - set(
                [
                    feature
                    for feature in VARIABLE_TYPE.keys()
                    if "categorical" in VARIABLE_TYPE[feature]
                ]
            )
        )

        # set the list of numeric and categorical features, to refer to in later methods
        self.numeric_features = numeric_features
        self.categorical_features = list(set(keepers) - set(numeric_features))

        return df

    def _prepare_numerical_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare numerical features.

        Parameters
        ----------
        df: pd.DataFrame
            The input data frame

        Returns
        -------
        np.ndarray
            The array of transformed numerical features.
        """
        df = df.copy()
        # for regression and neural networks, unskew those four features
        if self.model_type in [
            "RidgeRegressor",
            "MissingnessNeuralNetRegressor",
            "MissingnessNeuralNetClassifier",
            "MissingnessNeuralNetEIM",
        ]:
            for feature in ["CIG_0", "PRIORDEAD", "PRIORLIVE", "PRIORTERM", "RF_CESARN", "ILLB_R"]:
                df[feature] = np.log(df[feature] + 0.1)

        x_numeric = df[self.numeric_features].values

        # for the linear regression, we will decorrelate the continuous features
        if self.model_type == "RidgeRegressor":
            if self.standardization_parameters is None:
                self.standardization_parameters = {}
                self.standardization_parameters["means"] = x_numeric.mean(axis=0)
                q_matrix, r_matrix = np.linalg.qr(
                    x_numeric - self.standardization_parameters["means"]
                )
                self.standardization_parameters["r_matrix"] = r_matrix @ np.diag(
                    q_matrix.std(axis=0) * 3.0
                )

            x_numeric = (x_numeric - self.standardization_parameters["means"]) @ np.linalg.inv(
                self.standardization_parameters["r_matrix"]
            )

        # for the neural networks, we will standardize without decorrelating
        elif self.model_type in [
            "MissingnessNeuralNetRegressor",
            "MissingnessNeuralNetClassifier",
            "MissingnessNeuralNetEIM",
        ]:
            if self.standardization_parameters is None:
                self.standardization_parameters = {}
                self.standardization_parameters["means"] = np.nanmean(x_numeric, axis=0)
                self.standardization_parameters["sds"] = np.nanmax(
                    np.abs(x_numeric - np.nanmean(x_numeric, axis=0)), axis=0
                )

            x_numeric = (
                x_numeric - self.standardization_parameters["means"]
            ) / self.standardization_parameters["sds"]

        # tell the neural network how many missing points
        if self.model_type in [
            "MissingnessNeuralNetRegressor",
            "MissingnessNeuralNetClassifier",
            "MissingnessNeuralNetEIM",
        ]:
            num_missing = np.isnan(x_numeric).sum(axis=1).reshape(-1, 1)
            x_numeric = np.hstack([x_numeric, num_missing])

        return x_numeric

    def _prepare_categorical_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare categorical features.

        Parameters
        ----------
        df: pd.DataFrame
            The input data frame

        Returns
        -------
        np.ndarray
            The array of one-hot or integer encoded categorical features.
        """
        df = df.copy()
        if self.feature_categories is None:
            self.feature_categories = {}
            for feature in self.categorical_features:
                self.feature_categories[feature] = np.sort(df[feature].dropna().unique())

        # the tree-based models use integer-encoded categorical features
        if self.model_type in ["HistBoostRegressor", "WildWoodRegressor"]:
            x_integers_list = []
            for feature in self.categorical_features:
                integer_mapping = {
                    label: i for i, label in enumerate(self.feature_categories[feature])
                }
                x_integer_col = df[feature].map(integer_mapping).values

                # NaNs are encoded as -1
                x_integer_col = np.where(df[feature].isna(), -1, x_integer_col)
                x_integers_list.append(x_integer_col.reshape((-1, 1)))

            x_integers = np.hstack(x_integers_list)
            return x_integers

        else:
            x_one_hot_list = []
            for feature in self.categorical_features:
                # the OneHotEncoder raises warnings when encoding NaNs as zeros. It's noisy.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    x_one_hot_col = OneHotEncoder(
                        categories=[self.feature_categories[feature]],
                        sparse_output=False,
                        drop=None if self.model_type == "RidgeRegressor" else "first",
                        handle_unknown="ignore",
                    ).fit_transform(df[feature].values.reshape((-1, 1)))

                # put NAs back for the neural network
                if self.model_type in [
                    "MissingnessNeuralNetRegressor",
                    "MissingnessNeuralNetClassifier",
                    "MissingnessNeuralNetEIM",
                ]:
                    x_one_hot_col = x_one_hot_col * np.where(
                        df[feature].isna(), np.nan, 1.0
                    ).reshape((-1, 1))
                x_one_hot_list.append(x_one_hot_col)
            x_one_hot = np.hstack(x_one_hot_list)
            return x_one_hot

    def __call__(self, df: pd.DataFrame) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Prepare data.

        Parameters
        ----------
        df: pd.DataFrame
            The input data frame

        Returns
        -------
        Tuple or np.ndarray
            the design matrix X and the response variable y, if present.
        """
        df = df.copy()
        df = self._enforce_feature_types(df)
        df = self._process_missing_data(df)
        df = self._subset_and_binarize(df)
        X_numeric = self._prepare_numerical_features(df)
        X_categorical = self._prepare_categorical_features(df)

        # save the categorical feature mask to pass to the HistBoostRegressor
        self.categorical_features = np.concatenate(
            [np.zeros(X_numeric.shape[1]), np.ones(X_categorical.shape[1])]
        ).astype(bool)
        X = np.hstack([X_numeric, X_categorical])

        # For numerical stability, remove columns that are degenerate
        if self.model_type not in ["HistBoostRegressor", "WildWoodRegressor"]:
            if self.nondegenerate_columns is None:
                self.nondegenerate_columns = (X.max(axis=0) - X.min(axis=0)) != 0
            X = X[:, self.nondegenerate_columns]

        if "DBWT" in df.columns:
            # The ridge regressor will model the standardized weight
            if self.model_type in ["RidgeRegressor"]:
                y = (df["DBWT"].values.reshape((-1, 1)) - Y_MEAN) / Y_SD
            # The classifier will model binned weights
            elif self.model_type in ["MissingnessNeuralNetClassifier"]:
                bin_edges = np.concatenate(
                    [
                        np.array([-np.inf]),
                        (BIN_LABELS[1:] + BIN_LABELS[:-1]) / 2,
                        np.array([np.inf]),
                    ]
                )
                y = pd.cut(df["DBWT"], bins=bin_edges, retbins=False, labels=False).values
                self.bin_values = BIN_LABELS
            # the HistBoostRegressor will directly model the raw response variable
            elif self.model_type in ["HistBoostRegressor", "WildWoodRegressor"]:
                y = df["DBWT"].values
            # the neural networks will all work on a transformed scale (to enforce the lower
            # prediction interval to always be positive)
            else:
                y = np_softplus_inv(df["DBWT"].values.reshape((-1, 1)) / SOFTPLUS_SCALE)

            return X, y
        else:
            return X
