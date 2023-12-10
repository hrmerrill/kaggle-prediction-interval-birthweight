from datetime import datetime
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from kaggle_prediction_interval_birthweight.model.constants import MISSING_CODE, VARIABLE_TYPE

TIMESTAMP_COL_NAME = "DOB_TT"
Y_MEAN = 3260
Y_SD = 590

LOGY_MEAN = 8
LOGY_SD = 0.25


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
            One of "linear regression", "xgboost", or "neural network". Determines how data are
            imputed and standardized before modeling.
        standardization_parameters: Dict
            Precomputed columnwise means and sds or decorrelation matrix, if already computed.
            None during training, but at test time these values should be passed in.
        feature_categories: Dict
            levels of categorical variables. None during training (they are inferred), but at
            test time these values should be passed in.
        """
        allowed_model_types = ["linear regression", "xgboost", "neural network"]
        if model_type not in allowed_model_types:
            raise NotImplementedError(f"Supported model_type values: {allowed_model_types}")
        self.model_type = model_type
        self.standardization_parameters = standardization_parameters
        self.feature_categories = feature_categories

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
            The same data frame with categorical features as strings and numerical features as floats.
        """
        df = df.copy()

        # convert that timestamp column to an actual timestamp. First, as a string
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
        # for tree-based models, just use the missingness codes.
        if self.model_type == "xgboost":
            return df

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
            if self.model_type == "linear regression":
                if "categorical" in VARIABLE_TYPE[feature]:
                    replacement = df.loc[~df[feature].isin(missing_code), feature].mode()
                else:
                    replacement = df.loc[~df[feature].isin(missing_code), feature].median()

            # for the neural network, use real NAs with the expected relu layer.
            elif self.model_type == "neural network":
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
        # during the EDA, we decided to drop these
        keepers = list(set(VARIABLE_TYPE.keys()) - {"PAY", "ILP_R"})
        df = df.drop(["PAY", "ILP_R"], axis=1)

        # we also added a polynomial effect for one feature
        df["ILOP_R_2"] = df["ILOP_R"] ** 2
        df["ILOP_R_3"] = df["ILOP_R"] ** 3
        keepers = keepers + ["ILOP_R_2", "ILOP_R_3"]

        # NaNs and medians are taken care of, but for the tree-based model, use the missing code
        if self.model_type == "xgboost":
            df.loc[df["ILOP_R"].isin(MISSING_CODE["ILOP_R"]), "ILOP_R_2"] = MISSING_CODE["ILOP_R"][
                0
            ]
            df.loc[df["ILOP_R"].isin(MISSING_CODE["ILOP_R"]), "ILOP_R_3"] = MISSING_CODE["ILOP_R"][
                0
            ]

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
        # we also decided to binarize ILLB_R, so do that here, make it a string, and remove it from
        # the list of numeric features
        df["ILLB_R"] = (df["ILLB_R"] >= np.exp(2.75)).astype(str)
        numeric_features = list(set(numeric_features) - {"ILLB_R"})

        df["M_Ht_In"] = df["M_Ht_In"].clip(48, 78)

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
        # for regression and neural networks, unskew those four features
        if self.model_type in ["linear regression", "neural network"]:
            for feature in ["CIG_0", "PRIORDEAD", "PRIORTERM", "RF_CESARN"]:
                df[feature] = np.log(df[feature] + 1)

        x_numeric = df[self.numeric_features].values

        # for the linear regression, we will decorrelate the continuous features
        if self.model_type == "linear regression":
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

        # for the neural network, we will standardize without decorrelating
        elif self.model_type == "neural network":
            if self.standardization_parameters is None:
                self.standardization_parameters = {}
                self.standardization_parameters["means"] = np.nanmean(x_numeric, axis=0)
                self.standardization_parameters["sds"] = np.nanstd(x_numeric, axis=0) * 3.0

            x_numeric = (
                x_numeric - self.standardization_parameters["means"]
            ) / self.standardization_parameters["sds"]

        # tell the neural network where the missing data are
        if self.model_type == "neural network":
            x_numeric = np.hstack([x_numeric, np.where(np.isnan(x_numeric), 1, 0)])

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
            The array of one-hot encoded categorical features.
        """
        if self.feature_categories is None:
            self.feature_categories = {}
            for feature in self.categorical_features:
                self.feature_categories[feature] = np.sort(df[feature].dropna().unique())

        x_one_hot_list = []
        for feature in self.categorical_features:
            x_one_hot_col = OneHotEncoder(
                categories=[self.feature_categories[feature]],
                sparse_output=False,
                drop=None if self.model_type == "linear regression" else "first",
                handle_unknown="ignore",
            ).fit_transform(df[feature].values.reshape((-1, 1)))

            # put NAs back for the neural network
            if self.model_type == "neural network":
                x_one_hot_col = x_one_hot_col * np.where(df[feature].isna(), np.nan, 1.0).reshape(
                    (-1, 1)
                )
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
        df = self._enforce_feature_types(df)
        df = self._process_missing_data(df)
        df = self._subset_and_binarize(df)
        X_numeric = self._prepare_numerical_features(df)
        X_categorical = self._prepare_categorical_features(df)

        if "DBWT" in df.columns:
            if self.model_type == "linear regression":
                y = (df["DBWT"].values.reshape((-1, 1)) - Y_MEAN) / Y_SD
            else:
                y = (np.log(df["DBWT"]).values.reshape((-1, 1)) - LOGY_MEAN) / LOGY_SD

            return np.hstack([X_numeric, X_categorical]), y
        else:
            return np.hstack([X_numeric, X_categorical])