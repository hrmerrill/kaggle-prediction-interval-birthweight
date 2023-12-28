from typing import List, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import d2_pinball_score, make_scorer
from sklearn.model_selection import GridSearchCV

from kaggle_prediction_interval_birthweight.data.data_processing import SOFTPLUS_SCALE
from kaggle_prediction_interval_birthweight.model.sampling_utils import np_softplus


class HistBoostRegressor:
    """
    Extend the RidgeCV class to predict prediction intervals on the original scale.
    """

    def __init__(self, alpha: float = 0.9) -> None:
        """
        Parameters
        ----------
        alpha: float
            significance level for prediction intervals
        """
        param_grid = {
            "max_leaf_nodes": [20, None],
            "max_depth": [5, None],
            "min_samples_leaf": [10, 50],
            "l2_regularization": [0, 0.1],
            "learning_rate": [1, 0.1, 0.03],
        }
        self.lower_regressor = GridSearchCV(
            estimator=HistGradientBoostingRegressor(
                loss="quantile",
                quantile=(1 - alpha) / 2,
                max_iter=1000,
            ),
            param_grid=param_grid,
            scoring=make_scorer(lambda o, p: d2_pinball_score(o, p, alpha=(1 - alpha) / 2)),
            verbose=1,
            cv=3,
        )
        self.upper_regressor = GridSearchCV(
            estimator=HistGradientBoostingRegressor(
                loss="quantile",
                quantile=alpha + (1 - alpha) / 2,
                max_iter=1000,
            ),
            param_grid=param_grid,
            scoring=make_scorer(lambda o, p: d2_pinball_score(o, p, alpha=alpha + (1 - alpha) / 2)),
            verbose=1,
            cv=3,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the quantile boosting model.

        Parameters
        ----------
        X: np.ndarray
            Design matrix containing features
        y: np.ndarray
            Array of response values
        """
        self.lower_regressor.fit(X, y.squeeze())
        self.upper_regressor.fit(X, y.squeeze())

    def predict_intervals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the alpha * 100% interval for birthweight.

        Parameters
        ----------
        X: np.ndarray
            the design matrix used by self.fit(X, y)

        Returns
        -------
        Tuple
            the arrays corresponding to the lower and upper bounds, respectively
        """
        lower = np_softplus(self.lower_regressor.predict(X)) * SOFTPLUS_SCALE
        upper = np_softplus(self.upper_regressor.predict(X)) * SOFTPLUS_SCALE
        return lower.squeeze(), upper.squeeze()
