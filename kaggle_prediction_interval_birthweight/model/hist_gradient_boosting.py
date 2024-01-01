from typing import Optional, Tuple

import numpy as np
from mapie.regression import MapieQuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import d2_pinball_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split


class HistBoostRegressor:
    """
    Wrap around three histboost regressors.
    """

    def __init__(
        self, alpha: float = 0.9, categorical_feature_mask: Optional[np.ndarray] = None
    ) -> None:
        """
        Parameters
        ----------
        alpha: float
            significance level for prediction intervals
        categorical_feature_mask: np.ndarray
            optional boolean array indicating categorical features
        """
        self.alpha = alpha
        self.categorical_feature_mask = categorical_feature_mask
        param_grid = {
            "l2_regularization": [0, 1, 2],
            "learning_rate": [0.3, 0.4],
        }
        self.lower_regressor = GridSearchCV(
            estimator=HistGradientBoostingRegressor(
                loss="quantile",
                quantile=(1 - alpha) / 2,
                max_iter=1000,
                categorical_features=self.categorical_feature_mask,
                max_leaf_nodes=21,
                max_depth=4,
                min_samples_leaf=100,
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
                categorical_features=self.categorical_feature_mask,
                max_leaf_nodes=21,
                max_depth=4,
                min_samples_leaf=100,
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
                categorical_features=self.categorical_feature_mask,
                max_leaf_nodes=21,
                max_depth=4,
                min_samples_leaf=100,
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
        return lower.squeeze(), upper.squeeze()
