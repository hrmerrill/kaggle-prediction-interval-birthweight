from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import d2_pinball_score, make_scorer
from sklearn.model_selection import GridSearchCV

from kaggle_prediction_interval_birthweight.data.data_processing import DataProcessor
from kaggle_prediction_interval_birthweight.model.hist_gradient_boosting import HistBoostRegressor
from kaggle_prediction_interval_birthweight.model.linear_regression import RidgeRegressor
from kaggle_prediction_interval_birthweight.model.neural_network import MissingnessNeuralNet


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
        self.neural_networks = [MissingnessNeuralNet() for _ in range(n_folds)]
        self.lower_regressor = GridSearchCV(
            estimator=HistGradientBoostingRegressor(quantile=(1 - alpha) / 2, loss="quantile"),
            param_grid={"l2_regularization": [10**x for x in np.linspace(-4, 1, 15)]},
            scoring=make_scorer(lambda o, p: d2_pinball_score(o, p, alpha=(1 - alpha) / 2)),
            verbose=1,
        )
        self.upper_regressor = GridSearchCV(
            estimator=HistGradientBoostingRegressor(
                quantile=alpha + (1 - alpha) / 2, loss="quantile"
            ),
            param_grid={"l2_regularization": [10**x for x in np.linspace(-4, 1, 15)]},
            scoring=make_scorer(lambda o, p: d2_pinball_score(o, p, alpha=alpha + (1 - alpha) / 2)),
            verbose=1,
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
        self.nn_data_processor = DataProcessor(model_type="neural network")
        _ = self.ridge_data_processor(df)
        _ = self.boost_data_processor(df)
        _ = self.nn_data_processor(df)

        # create some information for holding new data in the data frame
        df["fold"] = np.random.choice(self.n_folds, df.shape[0])
        upstream_predictions = [
            "lower_ridge",
            "upper_ridge",
            "lower_boost",
            "upper_boost",
            "lower_nn",
            "upper_nn",
        ]
        for feature in upstream_predictions:
            df[feature] = None

        # loop across splits, fit and prediction from upstream models
        for k in range(self.n_folds):
            print(f"Ensembler fold {k+1} of {self.n_folds} begins.")

            # split into train and test sets
            df_train = df.query(f"fold != @k")
            df_test = df.query(f"fold == @k")

            # prepare the data for both ridge and histboost regression
            xr_train, yr_train = self.ridge_data_processor(df_train)
            xr_test = self.ridge_data_processor(df_test.drop("DBWT", axis=1))

            xb_train, yb_train = self.boost_data_processor(df_train)
            xb_test = self.boost_data_processor(df_test.drop("DBWT", axis=1))

            xn_train, yn_train = self.nn_data_processor(df_train)
            xn_test = self.nn_data_processor(df_test.drop("DBWT", axis=1))

            print("Training the ridge regression model.")
            self.ridge_regressors[k].fit(xr_train, yr_train)
            ridge_lower, ridge_upper = self.ridge_regressors[k].predict_intervals(
                xr_test, alpha=self.alpha
            )
            df.loc[df["fold"] == k, "lower_ridge"] = ridge_lower.squeeze()
            df.loc[df["fold"] == k, "upper_ridge"] = ridge_upper.squeeze()

            print("Training the histogram boosting model.")
            self.histboosters[k].fit(xb_train, yb_train)
            boost_lower, boost_upper = self.histboosters[k].predict_intervals(xb_test)
            df.loc[df["fold"] == k, "lower_boost"] = boost_lower.squeeze()
            df.loc[df["fold"] == k, "upper_boost"] = boost_upper.squeeze()

            print("Training the neural network model.")
            self.neural_networks[k].fit(xn_train, yn_train)
            nn_lower, nn_upper = self.neural_networks[k].predict_intervals(
                xn_test, alpha=self.alpha
            )
            df.loc[df["fold"] == k, "lower_nn"] = nn_lower.squeeze()
            df.loc[df["fold"] == k, "upper_nn"] = nn_upper.squeeze()

        print("Training the ensemble model.")

        # now train the ensembler on the left-out predictions from the previous two models
        x_ens = df[upstream_predictions].values
        y_ens = df["DBWT"].values.squeeze()

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
        xn = self.nn_data_processor(df.drop("DBWT", axis=1))

        lowers, uppers = [], []
        for k in range(self.n_folds):
            rl, ru = self.ridge_regressors[k].predict_intervals(xr)
            hl, hu = self.histboosters[k].predict_intervals(xb)
            nl, nu = self.neural_networks[k].predict_intervals(xn)
            x = np.hstack(
                [
                    rl.reshape((-1, 1)),
                    ru.reshape((-1, 1)),
                    hl.reshape((-1, 1)),
                    hu.reshape((-1, 1)),
                    nl.reshape((-1, 1)),
                    nu.reshape((-1, 1)),
                ]
            )
            lower = self.lower_regressor.predict(x)
            upper = self.upper_regressor.predict(x)
            lowers.append(lower)
            uppers.append(upper)

        lower = np.hstack([l.reshape((-1, 1)) for l in lowers]).mean(axis=1)
        upper = np.hstack([u.reshape((-1, 1)) for u in uppers]).mean(axis=1)
        return lower.squeeze(), upper.squeeze()
