from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import GridSearchCV

from kaggle_prediction_interval_birthweight.data.data_processing import DataProcessor
from kaggle_prediction_interval_birthweight.model.linear_regression import (
    RidgeRegressor,
)
from kaggle_prediction_interval_birthweight.model.hist_gradient_boosting import (
    HistBoostRegressor,
)


class Ensembler:
    """
    Create an ensemble model that combines the other models.
    """

    def __init__(self, n_folds: int = 3, alpha: float = 0.9) -> None:
        """
        Parameters
        ----------
        n_folds: int
            number of folds to use for held-out training
        alpha: float
            significance level for prediction intervals
        """
        self.alpha = alpha
        self.n_folds = n_folds
        self.histboosters = [HistBoostRegressor(alpha) for _ in range(n_folds)]
        self.ridge_regressors = [RidgeRegressor() for _ in range(n_folds)]
        self.lower_regressor = GridSearchCV(
            estimator=QuantileRegressor(quantile=(1 - alpha) / 2, solver="highs"),
            param_grid={"alpha": np.linspace(0, 2, 10)},
            scoring="neg_mean_squared_error",
        )
        self.upper_regressor = GridSearchCV(
            estimator=QuantileRegressor(quantile=alpha + (1 - alpha) / 2, solver="highs"),
            param_grid={"alpha": np.linspace(0, 2, 10)},
            scoring="neg_mean_squared_error",
        )

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the ensemble model.

        Parameters
        ----------
        df: pd.DataFrame
            The input data
        """
        df = df.copy()

        # save the data processors, so standardization parameters are available later
        self.ridge_data_processor = DataProcessor(model_type="linear regression")
        self.boost_data_processor = DataProcessor(model_type="xgboost")
        _ = self.ridge_data_processor(df)
        _ = self.boost_data_processor(df)

        # create some information for holding new data in the data frame
        df["fold"] = np.random.choice(self.n_folds, df.shape[0])
        upstream_predictions = ["lower_ridge", "upper_ridge", "lower_boost", "upper_boost"]
        for feature in upstream_predictions:
            df[feature] = None

        # loop across splits, fit and prediction from upstream models
        for k in range(self.n_folds):
            # split into train and test sets
            df_train = df.query(f"fold != @k")
            df_test = df.query(f"fold == @k")

            # prepare the data for both ridge and histboost regression
            xr_train, yr_train = self.ridge_data_processor(df_train)
            xr_test = self.ridge_data_processor(df_test.drop("DBWT", axis=1))

            xb_train, yb_train = self.boost_data_processor(df_train)
            xb_test = self.boost_data_processor(df_test.drop("DBWT", axis=1))

            self.ridge_regressors[k].fit(xr_train, yr_train)
            ridge_lower, ridge_upper = self.ridge_regressors[k].predict_intervals(
                xr_test, alpha=self.alpha
            )
            df.loc[df["fold"] == k, "lower_ridge"] = ridge_lower
            df.loc[df["fold"] == k, "upper_ridge"] = ridge_upper

            self.histboosters[k].fit(xb_train, yb_train)
            boost_lower, boost_upper = self.histboosters[k].predict_intervals(xb_test)
            df.loc[df["fold"] == k, "lower_boost"] = boost_lower
            df.loc[df["fold"] == k, "upper_boost"] = boost_upper

        # now train the ensembler on the left-out predictions from the previous two models
        x_ens = df[upstream_predictions].values
        y_ens = df["DBWT"].values

        self.lower_regressor.fit(x_ens, y_ens)
        self.upper_regressor.fit(x_ens, y_ens)

    def predict_intervals(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the alpha * 100% interval for birthweight.

        Parameters
        ----------
        df: pd.DataFrame
            The input data

        Returns
        -------
        Tuple
            the arrays corresponding to the lower and upper bounds, respectively
        """
        df = df.copy()

        xr = self.ridge_data_processor(df.drop("DBWT", axis=1))
        xb = self.boost_data_processor(df.drop("DBWT", axis=1))

        lowers, uppers = [], []
        for k in range(self.n_folds):
            rl, ru = self.ridge_regressors[k].predict_intervals(xr)
            hl, hu = self.histboosters[k].predict_intervals(xb)
            x = np.hstack(
                [
                    rl.reshape((-1, 1)),
                    ru.reshape((-1, 1)),
                    hl.reshape((-1, 1)),
                    hu.reshape((-1, 1)),
                ]
            )
            lower = self.lower_regressor.predict(x)
            upper = self.upper_regressor.predict(x)
            lowers.append(lower)
            uppers.append(upper)

        lower = np.hstack([l.reshape((-1, 1)) for l in lowers]).mean(axis=0)
        upper = np.hstack([u.reshape((-1, 1)) for l in uppers]).mean(axis=0)
        return lower.squeeze(), upper.squeeze()
