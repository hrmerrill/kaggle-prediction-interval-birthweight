from typing import Tuple

import numpy as np
import pandas as pd

import kaggle_prediction_interval_birthweight.model as kaggle_models
from kaggle_prediction_interval_birthweight.data.data_processing import DataProcessor
from kaggle_prediction_interval_birthweight.workflow.metric import score


class Validator:
    """Class containing methods for cross-validation."""

    def __init__(self, model_type: str, n_folds: int = 2, **kwargs) -> None:
        """
        Parameters
        ----------
        model_type: str
            one of RidgeRegressor, HistBoostRegressor, MissingnessNeuralNetRegressor,
            MissingnessNeuralNetEIM, MissingnessNeuralNetClassifier,
            HistBoostEnsembler, or NeuralNetEnsembler
        n_folds: int
            number of folds over which to cross-validate
        **kwargs: dict
            additional arguments passed to model_type
        """
        self.model_type = model_type
        self.n_folds = n_folds
        self.kwargs = kwargs
        if model_type not in ["HistBoostEnsembler", "NeuralNetEnsembler"]:
            self.data_processors = [DataProcessor(model_type) for _ in range(n_folds)]
        self.models = [
            getattr(kaggle_models, self.model_type)(**self.kwargs) for _ in range(self.n_folds)
        ]

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the models to each fold.

        Parameters
        ----------
        df: pd.DataFrame
            the input data
        """
        df = df.copy()
        df["cv_fold"] = np.random.choice(self.n_folds, df.shape[0])

        lower_list, upper_list, observations_list = [], [], []
        for cv_fold in range(self.n_folds):
            print(f"Validation on fold {cv_fold+1} of {self.n_folds} begins.")

            df_train = df.query("cv_fold != @cv_fold")
            df_test = df.query("cv_fold == @cv_fold")

            if self.model_type not in ["HistBoostEnsembler", "NeuralNetEnsembler"]:
                x_train, y_train = self.data_processors[cv_fold](df_train)
                x_test, y_test = self.data_processors[cv_fold](df_test)
                self.models[cv_fold].fit(x_train, y_train)
                lower, upper = self.models[cv_fold].predict_intervals(x_test)
            else:
                self.models[cv_fold].fit(df_train)
                lower, upper = self.models[cv_fold].predict_intervals(df_test)

            lower_list.append(lower)
            upper_list.append(upper)
            observations_list.append(df_test["DBWT"].values)

        self.lower_bounds = np.concatenate(lower_list)
        self.upper_bounds = np.concatenate(upper_list)
        self.observations = np.concatenate(observations_list)

    def compute_performance(self) -> Tuple[float, float]:
        """Compute the NOIS and coverage on cross-validated data."""
        nois, coverage = score(self.observations, self.lower_bounds, self.upper_bounds, 0.9)
        return nois, coverage

    def print_performance_summary(self) -> None:
        """Print a summary of performance."""
        nois, coverage = self.compute_performance()
        print(f"NOIS is {nois:.1f} and coverage is {coverage*100:.1f}%.")

    def predict_intervals(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the alpha * 100% interval for birthweight.

        Parameters
        ----------
        df: pd.DataFrame
            The input data

        Returns
        -------
        Tuple
            the arrays corresponding to the lower and upper bounds, respectively
        """
        df = df.copy()
        if "DBWT" in df.columns:
            df = df.drop("DBWT", axis=1)

        lowers, uppers = [], []
        for cv_fold in range(self.n_folds):
            if self.model_type in ["HistBoostEnsembler", "NeuralNetEnsembler"]:
                lower, upper = self.models[cv_fold].predict_intervals(df)
            else:
                x = self.data_processors[cv_fold](df)
                lower, upper = self.models[cv_fold].predict_intervals(x)
            lowers.append(lower.reshape((-1, 1)))
            uppers.append(upper.reshape((-1, 1)))

        return np.hstack(lowers).mean(axis=1), np.hstack(uppers).mean(axis=1)
