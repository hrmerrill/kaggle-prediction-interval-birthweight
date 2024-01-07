from typing import List, Optional, Union

import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from wildwood import ForestRegressor

BASE_PARAMETER_SPACE = [
    Integer(30, 50, name="n_estimators"),
    Real(10, 100, name="step"),
]

# skopt has still not fixed this bug.
np.int = np.int64


class WildWoodTuner:
    """
    Tune the wildwood regressor model.
    """

    def __init__(
        self,
        parameter_space: List[Union[Real, Integer]] = BASE_PARAMETER_SPACE,
        categorical_feature_mask: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        parameter_space: List
            the parameter space to search
        categorical_feature_mask: np.ndarray
            optional boolean array indicating categorical features
        verbose: bool
            verbosity of the optimizer
        """
        self.parameter_space = parameter_space
        self.categorical_feature_mask = categorical_feature_mask
        self.verbose = verbose

    def tune(self, X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> None:
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
        """
        fold_ids = np.random.choice(n_folds, X.shape[0])

        @use_named_args(self.parameter_space)
        def objective(**params):
            print(params)
            scores = []
            for fold in range(n_folds):
                # get a training set and a test set
                xtr = X[fold_ids != fold]
                ytr = y[fold_ids != fold]
                xtest = X[fold_ids == fold]
                ytest = y[fold_ids == fold]

                # all three models get the same parameters
                regressor = ForestRegressor(
                    categorical_features=self.categorical_feature_mask,
                    n_estimators=int(params["n_estimators"]),
                    step=float(params["step"]),
                    max_depth=None,
                    min_samples_leaf=1,
                )
                regressor.fit(xtr, ytr.squeeze())
                predictions = regressor.predict(xtest)
                scores.append(np.sqrt(np.mean((predictions.squeeze() - ytest.squeeze()) ** 2)))

            return np.mean(scores)

        self.res_gp = gp_minimize(
            objective,
            self.parameter_space,
            n_calls=20,
            random_state=1,
            acq_func="EI",
            verbose=self.verbose,
        )
        self.result = {
            "score": self.res_gp.fun,
            "opt_parameters": {
                "n_estimators": self.res_gp.x[0],
                "step": self.res_gp.x[1],
            },
        }
