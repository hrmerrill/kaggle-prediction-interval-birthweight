from typing import List, Tuple

import numpy as np
import scipy.stats as st
from sklearn.linear_model import RidgeCV

from kaggle_prediction_interval_birthweight.data.data_processing import Y_MEAN, Y_SD


class RidgeRegressor(RidgeCV):
    """
    Extend the RidgeCV class to predict prediction intervals on the original scale.
    """

    def __init__(self, alphas: List[float] = [10**x for x in np.linspace(-3, 0, 20)]) -> None:
        """
        Parameters
        ----------
        alphas: List
            list of penalty parameters over which to search
        """
        super(RidgeRegressor, self).__init__(alphas=alphas, scoring="neg_mean_squared_error")

    def predict_intervals(self, X: np.ndarray, alpha: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the alpha * 100% interval for birthweight.

        Parameters
        ----------
        X: np.ndarray
            the design matrix used by self.fit(X, y)
        alpha: float
            the confidence level of the desired prediction interval

        Returns
        -------
        Tuple
            the arrays corresponding to the lower and upper bounds, respectively
        """
        predictions = self.predict(X)
        half_width = np.sqrt(-self.best_score_) * st.norm.ppf(alpha + (1 - alpha) / 2)
        lower = (predictions - half_width) * Y_SD + Y_MEAN
        upper = (predictions + half_width) * Y_SD + Y_MEAN
        return lower.squeeze(), upper.squeeze()
