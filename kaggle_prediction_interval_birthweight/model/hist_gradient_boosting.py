from typing import Tuple

import numpy as np
from mapie.regression import MapieQuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import d2_pinball_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split

from kaggle_prediction_interval_birthweight.data.data_processing import SOFTPLUS_SCALE
from kaggle_prediction_interval_birthweight.model.utils import np_softplus


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
        self.alpha = alpha
        param_grid = {
            "l2_regularization": [0, 0.1],
            "learning_rate": [1, 0.1, 0.01],
        }
        self.lower_regressor = GridSearchCV(
            estimator=HistGradientBoostingRegressor(
                loss="quantile",
                quantile=(1 - alpha) / 2,
                max_iter=1000,
                max_leaf_nodes=None,
                max_depth=None,
                min_samples_leaf=10,
            ),
            param_grid=param_grid,
            scoring=make_scorer(lambda o, p: d2_pinball_score(o, p, alpha=(1 - alpha) / 2)),
            verbose=1,
        )
        self.upper_regressor = GridSearchCV(
            estimator=HistGradientBoostingRegressor(
                loss="quantile",
                quantile=alpha + (1 - alpha) / 2,
                max_iter=1000,
                max_leaf_nodes=None,
                max_depth=None,
                min_samples_leaf=10,
            ),
            param_grid=param_grid,
            scoring=make_scorer(lambda o, p: d2_pinball_score(o, p, alpha=alpha + (1 - alpha) / 2)),
            verbose=1,
        )
        self.median_regressor = GridSearchCV(
            estimator=HistGradientBoostingRegressor(
                quantile=0.5,
                loss="quantile",
                max_iter=1000,
                max_leaf_nodes=50,
                max_depth=None,
                min_samples_leaf=20,
            ),
            param_grid=param_grid,
            scoring=make_scorer(lambda o, p: d2_pinball_score(o, p, alpha=0.5)),
            verbose=1,
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
        xtr, xval, ytr, yval = train_test_split(X, y, random_state=1, test_size=0.3)
        self.median_regressor.fit(xtr, ytr.squeeze())
        self.lower_regressor.fit(xtr, ytr.squeeze())
        self.upper_regressor.fit(xtr, ytr.squeeze())
        print("Calibrating with Mapie.")
        self.calibrator = MapieQuantileRegressor(
            [
                self.lower_regressor.best_estimator_,
                self.upper_regressor.best_estimator_,
                self.median_regressor.best_estimator_,
            ],
            alpha=1 - self.alpha,
            cv="prefit",
        )
        self.calibrator.fit(xval, yval.squeeze())

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
        _, intervals = self.calibrator.predict(X)
        lower, upper = intervals.squeeze().T
        lower = np_softplus(lower) * SOFTPLUS_SCALE
        upper = np_softplus(upper) * SOFTPLUS_SCALE
        return lower.squeeze(), upper.squeeze()
