from typing import List, Union

import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

import kaggle_prediction_interval_birthweight.model.neural_network as nns

BASE_PARAMETER_SPACE = [
    Real(0.05, 0.95, name="dropout_rate"),
    Integer(1, 5, name="n_layers"),
    Integer(5, 800, name="n_nodes_per_layer"),
]

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

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 3,
        model_type: str = "MissingnessNeuralNetRegressor",
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
        model_type: str
            One of MissingnessNeuralNetClassifier, MissingnessNeuralNetEIM, or
            MissingnessNeuralNetRegressor
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

                model = getattr(nns, model_type)(
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
        )
        self.result = {
            "score": self.res_gp.fun,
            "opt_parameters": {
                "dropout_rate": self.res_gp.x[0],
                "n_layers": self.res_gp.x[1],
                "n_nodes_per_layer": self.res_gp.x[2],
            },
        }
