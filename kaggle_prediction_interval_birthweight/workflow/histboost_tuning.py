from typing import List, Optional, Union

import numpy as np
from mapie.regression import MapieQuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import d2_pinball_score
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

from kaggle_prediction_interval_birthweight.workflow.metric import score

BASE_PARAMETER_SPACE = [
    Real(10**-3, 10**0, name="learning_rate"),
    Real(10**-5, 10**1, name="l2_regularization"),
    Integer(1, 10, name="max_depth"),
    Integer(5, 50, name="max_leaf_nodes"),
    Integer(10, 200, name="min_samples_leaf"),
]

SAVED_PRIOR_RUNS = {
    "y0": [
        -0.19476483,
        -0.21084902,
        -0.19296199,
        -0.20374554,
        -0.20260468,
        -0.20521543,
        -0.21441505,
        -0.21411501,
        -0.2090968,
        -0.21109029,
    ],
    "x0": [
        [0.5932517736067934, 8.44265904315269, 9, 43, 128],
        [0.3849973255854072, 2.975353090098658, 2, 17, 101],
        [0.8123565600467179, 4.79977692397885, 5, 43, 74],
        [0.6485237001791461, 3.682421715990082, 10, 11, 175],
        [0.4741344372284369, 8.009109510688925, 6, 36, 147],
        [0.5824377722830322, 5.373736920757813, 8, 10, 100],
        [0.18714601098343323, 7.369184402107811, 3, 11, 72],
        [0.15052519231649952, 2.223221659301995, 4, 46, 95],
        [0.6134503944262484, 9.023486808254013, 2, 49, 134],
        [0.17173867555090913, 3.5815280881735814, 8, 32, 72],
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

    def tune(
        self, X: np.ndarray, y: np.ndarray, n_folds: int = 5, include_mapie: bool = False
    ) -> None:
        """
        Tune the quantile boosting model.

        Parameters
        ----------
        X: np.ndarray
            Design matrix containing features
        y: np.ndarray
            Array of response values
        n_folds: int
            Number of folds for cross-validation
        include_mapie: bool
            If True, predictions are calibrated and NOIS is the returned score.
            If False, the average pinball loss directly from the regressors is the score.
        """
        fold_ids = np.random.choice(n_folds, X.shape[0])

        @use_named_args(self.parameter_space)
        def objective(**params):
            scores = []
            for fold in range(n_folds):
                # get a training set and a test set
                xtr = X[fold_ids != fold]
                ytr = y[fold_ids != fold]
                xtest = X[fold_ids == fold]
                ytest = y[fold_ids == fold]

                # a calibration set should be included if Mapie is used during tuning
                if include_mapie:
                    xtr, xval, ytr, yval = train_test_split(
                        xtr, ytr, random_state=fold, test_size=0.3
                    )

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

                # if using Mapie, the objective to minimize is NOIS
                if include_mapie:
                    calibrator = MapieQuantileRegressor(
                        [lower_regressor, upper_regressor, median_regressor],
                        alpha=1 - self.alpha,
                        cv="prefit",
                    )
                    calibrator.fit(xval, yval.squeeze())

                    _, intervals = calibrator.predict(xtest)
                    lower, upper = intervals.squeeze().T
                    nois, _ = score(ytest.squeeze(), lower, upper, self.alpha)
                    scores.append(nois)
                # Otherwise, we'll minimize the average quantile losses
                else:
                    pred_median = median_regressor.predict(xtest)
                    pred_lower = lower_regressor.predict(xtest)
                    pred_upper = upper_regressor.predict(xtest)

                    scores.append(-d2_pinball_score(ytest.squeeze(), pred_median, alpha=0.5))
                    scores.append(
                        -d2_pinball_score(ytest.squeeze(), pred_lower, alpha=(1 - self.alpha) / 2)
                    )
                    scores.append(
                        -d2_pinball_score(
                            ytest.squeeze(), pred_upper, alpha=self.alpha + (1 - self.alpha) / 2
                        )
                    )

            return np.mean(scores)

        self.res_gp = gp_minimize(
            objective,
            self.parameter_space,
            n_calls=50,
            random_state=1,
            acq_func="EI",
            verbose=self.verbose,
            y0=SAVED_PRIOR_RUNS["y0"],
            x0=SAVED_PRIOR_RUNS["x0"],
            n_jobs=-1,
        )
        self.result = {
            "score": self.res_gp.fun,
            "opt_parameters": {
                "learning_rate": self.res_gp.x[0],
                "l2_regularization": self.res_gp.x[1],
                "max_depth": self.res_gp.x[2],
                "max_leaf_nodes": self.res_gp.x[3],
                "min_samples_leaf": self.res_gp.x[4],
            },
        }
