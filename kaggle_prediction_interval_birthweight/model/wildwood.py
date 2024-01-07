from typing import Optional, Tuple

import numpy as np
import scipy.stats as st
from mapie.regression import MapieQuantileRegressor
from sklearn.model_selection import train_test_split
from wildwood import ForestRegressor


class MapieHelper:
    """Helper class for using Mapie for calibration of predicted intervals."""

    def __init__(self, model: ForestRegressor, which_quantile: float, sigma_sq: float) -> None:
        """
        Parameters
        ----------
        model: ForestRegressor
            The trained wildwood forest regressor
        which_quantile: int
            The quantile from the trees to predict
        sigma_sq: float
            Residual standard deviation
        """
        self.model = model
        self.which_quantile = which_quantile
        self.sigma_sq = sigma_sq
        self.__sklearn_is_fitted__ = lambda: True

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Doesn't do anything, just required by Mapie."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict from the model."""
        predictions = self.model.predict(X).squeeze()
        half_width = self.sigma_sq * st.norm.ppf(self.which_quantile)
        return predictions + half_width


class WildWoodRegressor:
    """
    Wrap around a wildwood regressor.
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
        self.regressor = ForestRegressor(
            categorical_features=self.categorical_feature_mask,
            n_estimators=50,
            step=100.0,
            max_depth=None,
            min_samples_leaf=1,
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
        self.regressor.fit(xtr, ytr.squeeze())
        sigma_sq = np.mean((yval.squeeze() - self.regressor.predict(xval).squeeze()) ** 2)

        print("Calibrating with Mapie.")
        self.calibrator = MapieQuantileRegressor(
            [
                MapieHelper(model=self.regressor, which_quantile=q, sigma_sq=sigma_sq)
                for q in [(1 - self.alpha) / 2, self.alpha + (1 - self.alpha) / 2, 0.5]
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
