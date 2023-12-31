from typing import List, Optional, Union

import numpy as np
from mapie.regression import MapieQuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

from kaggle_prediction_interval_birthweight.workflow.metric import score

BASE_PARAMETER_SPACE = [
    Real(10**-3, 10**0, "log-uniform", name="learning_rate"),
    Real(10**-5, 10**0, "log-uniform", name="l2_regularization"),
    Integer(1, 10, name="max_depth"),
    Integer(5, 50, name="max_leaf_nodes"),
    Integer(10, 100, name="min_samples_leaf"),
]

SAVED_PRIOR_RUNS = {
    "y0": [
        1681.11592994,
        1677.99559318,
        1699.64182955,
        1665.34872695,
        1674.25537926,
        1723.968289,
        1664.06507615,
        1771.68227481,
        1715.00671317,
        1682.98376227,
    ],
    "x0": [
        [0.9807412289900471, 0.46003068044903095, 2, 50, 31],
        [0.015478974396671025, 0.0008700690210600545, 7, 47, 86],
        [0.008706037847015437, 0.004195085318955972, 5, 15, 58],
        [0.5519326346081447, 0.0019320752621429876, 5, 47, 80],
        [0.140576118573469, 0.10322562466964824, 2, 28, 88],
        [0.30721380131411513, 0.1406101906115138, 3, 8, 70],
        [0.06014458808993631, 0.022817627560812234, 5, 14, 36],
        [0.0026690727196422457, 0.08252249755325355, 5, 7, 66],
        [0.09591956282799739, 0.0003107967535980341, 5, 15, 17],
        [0.02556910810331702, 3.0259468970108244e-05, 9, 10, 57],
    ],
}

# skopt has still not fixed this bug.
np.int = np.int64


class HistBoostTuner:
    """
    Tune the histboost regressor model.
    """

    def __init__(
        self,
        alpha: float = 0.9,
        parameter_space: List[Union[Real, Integer]] = BASE_PARAMETER_SPACE,
        categorical_feature_mask: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        alpha: float
            significance level for prediction intervals
        parameter_space: List
            the parameter space to search
        categorical_feature_mask: np.ndarray
            optional boolean array indicating categorical features
        verbose: bool
            verbosity of the optimizer
        """
        self.alpha = alpha
        self.parameter_space = parameter_space
        self.categorical_feature_mask = categorical_feature_mask
        self.verbose = verbose

    def tune(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Tune the quantile boosting model.

        Parameters
        ----------
        X: np.ndarray
            Design matrix containing features
        y: np.ndarray
            Array of response values
        """

        @use_named_args(self.parameter_space)
        def objective(**params):
            # get a training set and a test set
            xtr, xtest, ytr, ytest = train_test_split(X, y, random_state=1, test_size=0.3)

            # now split the training set into train and calibration
            xtr, xval, ytr, yval = train_test_split(xtr, ytr, random_state=1, test_size=0.3)

            # all three models get the same parameters
            lower_regressor = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=(1 - self.alpha) / 2,
                max_iter=1000,
                categorical_features=self.categorical_feature_mask,
                **params,
            )
            upper_regressor = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=self.alpha + (1 - self.alpha) / 2,
                max_iter=1000,
                categorical_features=self.categorical_feature_mask,
                **params,
            )
            median_regressor = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=0.5,
                max_iter=1000,
                categorical_features=self.categorical_feature_mask,
                **params,
            )
            median_regressor.fit(xtr, ytr.squeeze())
            lower_regressor.fit(xtr, ytr.squeeze())
            upper_regressor.fit(xtr, ytr.squeeze())

            calibrator = MapieQuantileRegressor(
                [lower_regressor, upper_regressor, median_regressor],
                alpha=1 - self.alpha,
                cv="prefit",
            )
            calibrator.fit(xval, yval.squeeze())

            _, intervals = calibrator.predict(xtest)
            lower, upper = intervals.squeeze().T
            nois, _ = score(ytest.squeeze(), lower, upper, self.alpha)
            return nois

        self.res_gp = gp_minimize(
            objective,
            self.parameter_space,
            n_calls=50,
            random_state=1,
            acq_func="EI",
            verbose=self.verbose,
            y0=SAVED_PRIOR_RUNS["y0"],
            x0=SAVED_PRIOR_RUNS["x0"],
        )
        self.result = {
            "NOIS": self.res_gp.fun,
            "opt_parameters": {
                "learning_rate": self.res_gp.x[0],
                "l2_regularization": self.res_gp.x[1],
                "max_depth": self.res_gp.x[2],
                "max_leaf_nodes": self.res_gp.x[3],
                "min_samples_leaf": self.res_gp.x[4],
            },
        }
