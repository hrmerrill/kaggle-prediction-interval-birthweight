from typing import List, Union

import numpy as np
from mapie.regression import MapieQuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

from kaggle_prediction_interval_birthweight.data.data_processing import SOFTPLUS_SCALE
from kaggle_prediction_interval_birthweight.utils.utils import np_softplus
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
        1653.28715,
        1683.15850331,
        1672.91563387,
        1687.35993538,
        1665.84357978,
        1692.47181337,
        1641.53385425,
        1677.72027894,
        1767.31711474,
        1653.6255668,
        1678.08465425,
        1768.72853743,
        1670.67686558,
        1771.41064552,
        1679.23823283,
        1672.98217349,
        1741.11285539,
        1779.85086934,
        1663.65901165,
        1671.56059291,
        1665.06594957,
    ],
    "x0": [
        [0.30721, 0.14061, 3, 8, 70],
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
        [1.0, 9.373116718663044e-05, 8, 40, 73],
        [0.19229648591368678, 0.0025734370536926016, 5, 8, 14],
        [0.6319077346026674, 1.0, 1, 9, 100],
        [0.6281223626435, 0.5825097940753016, 3, 34, 80],
        [0.6540089853140347, 0.018693766714316707, 6, 19, 16],
        [0.01033192611931315, 1.6868405139322896e-05, 3, 5, 18],
        [1.0, 2.2192365845278226e-05, 3, 12, 71],
        [0.3121546947692833, 0.05957328587483314, 9, 37, 77],
        [0.21966002067321405, 2.0319969909620314e-05, 9, 50, 65],
        [0.12277796061245785, 1.5166349577164813e-05, 10, 44, 26],
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
        verbose: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        alpha: float
            significance level for prediction intervals
        parameter_space: List
            the parameter space to search
        verbose: bool
            verbosity of the optimizer
        """
        self.alpha = alpha
        self.parameter_space = parameter_space
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
                **params,
            )
            upper_regressor = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=self.alpha + (1 - self.alpha) / 2,
                max_iter=1000,
                **params,
            )
            median_regressor = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=0.5,
                max_iter=1000,
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
            lower = np_softplus(lower) * SOFTPLUS_SCALE
            upper = np_softplus(upper) * SOFTPLUS_SCALE
            nois, _ = score(np_softplus(ytest.squeeze()) * SOFTPLUS_SCALE, lower, upper, self.alpha)
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
