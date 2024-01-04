from typing import List, Union

import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

from kaggle_prediction_interval_birthweight.model.neural_network import (
    MissingnessNeuralNetRegressor,
)

BASE_PARAMETER_SPACE = [
    Real(0.05, 0.95, name="dropout_rate"),
    Integer(1, 3, name="n_layers"),
    Integer(5, 600, name="n_nodes_per_layer"),
]

SAVED_PRIOR_RUNS = {
    "y0": [
        0.69154839,
        0.69508183,
        0.70045203,
        0.69236992,
        0.6995335,
        0.7559,
        0.7339,
        0.69498477,
        0.69045524,
        0.70065824,
        0.69114637,
        0.68820417,
        0.69411244,
        0.70939726,
        0.68997242,
        0.69070683,
        0.69691994,
        0.69473422,
        0.6914115,
        1.20096195,
        0.69356787,
        0.69164614,
        0.69371935,
        0.8235173,
        35.26728068,
        0.71675567,
        0.76874232,
        0.70602852,
        0.72567467,
        0.71931992,
        0.69914019,
        0.69867265,
    ],
    "x0": [
        [0.6993763349606942, 2, 400],
        [0.6001020958459112, 2, 527],
        [0.33824438095671183, 2, 537],
        [0.6440753159765015, 3, 434],
        [0.5717923775592872, 1, 521],
        [0.9060878595993938, 1, 5],
        [0.6603084771373383, 1, 5],
        [0.5835601564025166, 3, 515],
        [0.8125265649057131, 2, 234],
        [0.31778114589002504, 1, 167],
        [0.47989860558921493, 3, 291],
        [0.4035063164907468, 3, 206],
        [0.6333546848460776, 2, 575],
        [0.17631570237138067, 3, 287],
        [0.77081967678168, 2, 409],
        [0.6985693892533251, 2, 325],
        [0.7327540618901216, 1, 287],
        [0.05, 3, 5],
        [0.95, 3, 600],
        [0.95, 3, 310],
        [0.8073418269158793, 3, 600],
        [0.48590071864411416, 3, 333],
        [0.7921562751222541, 3, 39],
        [0.05, 3, 600],
        [0.19179299855305515, 3, 5],
        [0.1995299219024077, 1, 449],
        [0.05, 3, 288],
        [0.3114267185399016, 1, 311],
        [0.13546957625539247, 3, 379],
        [0.52021364534323, 1, 5],
        [0.43749335634791475, 3, 600],
        [0.5829927871300723, 1, 186],
    ],
}

# skopt has still not fixed this bug.
np.int = np.int64


class NeuralNetTuner:
    """
    Tune the neural network regressor model.
    """

    def __init__(
        self,
        parameter_space: List[Union[Real, Integer]] = BASE_PARAMETER_SPACE,
        verbose: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        parameter_space: List
            the parameter space to search
        verbose: bool
            verbosity of the optimizer
        """
        self.parameter_space = parameter_space
        self.verbose = verbose

    def tune(self, X: np.ndarray, y: np.ndarray, n_folds: int = 3) -> None:
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

                model = MissingnessNeuralNetRegressor(
                    units_list=[int(params["n_nodes_per_layer"])] * int(params["n_layers"]),
                    dropout_rate=params["dropout_rate"],
                )
                model.fit(xtr, ytr)
                scores.append(model.model.evaluate(xtest, ytest))

            return np.mean(scores)

        self.res_gp = gp_minimize(
            objective,
            self.parameter_space,
            n_calls=50,
            random_state=0,
            acq_func="EI",
            verbose=self.verbose,
            y0=SAVED_PRIOR_RUNS["y0"],
            x0=SAVED_PRIOR_RUNS["x0"],
        )
        self.result = {
            "score": self.res_gp.fun,
            "opt_parameters": {
                "dropout_rate": self.res_gp.x[0],
                "n_layers": self.res_gp.x[1],
                "n_nodes_per_layer": self.res_gp.x[2],
            },
        }
