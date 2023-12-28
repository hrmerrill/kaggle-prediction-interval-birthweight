from typing import Optional

import pandas as pd
import numpy as np
import typer

from kaggle_prediction_interval_birthweight.model.hist_gradient_boosting import HistBoostRegressor
from kaggle_prediction_interval_birthweight.model.neural_network import (
    SOFTPLUS_SCALE,
    MissingnessNeuralNetRegressor,
)
from kaggle_prediction_interval_birthweight.model.sampling_utils import np_softplus_inv
from kaggle_prediction_interval_birthweight.workflow.validation import Validator

LOCAL_DIR = "~/dev/data/kaggle-prediction-interval-birthweight/"


app = typer.Typer()


@app.command()
def create_submission(
    model_type: str,
    train_data_path: str = LOCAL_DIR + "train.csv",
    test_data_path: str = LOCAL_DIR + "test.csv",
    output_path: Optional[str] = None,
) -> None:
    """
    Create a submission for the competition.

    Parameters
    ----------
    model_type: str
        The type of model to use. One of RidgeRegressor, HistBoostRegressor,
        MissingnessNeuralNetRegressor, MissingnessNeuralNetClassifier,
        MissingnessNeuralNetEIM, HistBoostEnsembler, or NeuralNetEnsembler
    train_data_path: str
        path to training data
    test_data_path: str
        path to test data
    output_path: str
        path to save the output file for submission to kaggle
    """
    if output_path is None:
        output_path = LOCAL_DIR + f"submission_{model_type}.csv"

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    validator = Validator(model_type)
    validator.fit(train_data)
    validator.print_performance_summary()

    lower, upper = validator.predict_intervals(test_data)
    test_data[["id"]].assign(pi_lower=lower, pi_upper=upper).to_csv(output_path, index=False)
    print("Submission file saved to: \n" + output_path)


@app.command()
def create_hail_mary_submission(
    train_data_path: str = LOCAL_DIR + "train.csv",
    test_data_path: str = LOCAL_DIR + "test.csv",
) -> None:
    """
    Create a submission that ensembles all model types.

    Parameters
    ----------
    train_data_path: str
        path to training data
    test_data_path: str
        path to test data
    """
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    all_types = [
        "RidgeRegressor",
        "HistBoostRegressor",
        "MissingnessNeuralNetRegressor",
        "MissingnessNeuralNetClassifier",
        "MissingnessNeuralNetEIM",
        "HistBoostEnsembler",
        "NeuralNetEnsembler",
    ]
    train_ensemble, test_ensemble = [], []
    for model_type in all_types:
        print(f"Beginning training the {model_type}:")
        validator = Validator(model_type)
        validator.fit(train_data)

        print(f"{model_type} performance:")
        validator.print_performance_summary()

        lower_train, upper_train = validator.predict_intervals(train_data)
        lower_test, upper_test = validator.predict_intervals(test_data)

        test_data[["id"]].assign(pi_lower=lower_test, pi_upper=upper_test).to_csv(
            LOCAL_DIR + f"submission_{model_type}.csv", index=False
        )
        print("Submission file saved to: \n" + LOCAL_DIR + f"submission_{model_type}.csv")

        train_ensemble + [lower_train / SOFTPLUS_SCALE, upper_train / SOFTPLUS_SCALE]
        test_ensemble + [lower_test / SOFTPLUS_SCALE, upper_test / SOFTPLUS_SCALE]

        print(f"{model_type} complete.")

    train_ensemble = np.hstack([x.reshape((-1, 1)) for x in train_ensemble])
    test_ensemble = np.hstack([x.reshape((-1, 1)) for x in test_ensemble])
    train_y = np_softplus_inv(train_data["DBWT"] / SOFTPLUS_SCALE)

    print("Training the hail mary neural network.")
    model_nn = MissingnessNeuralNetRegressor(bayesian=True, fit_tail=False)
    model_nn.fit(train_ensemble, train_y)
    lower_nn, upper_nn = model_nn.predict_intervals(test_ensemble, alpha=0.9, n_samples=2000)

    test_data[["id"]].assign(pi_lower=lower_nn, pi_upper=upper_nn).to_csv(
        LOCAL_DIR + f"submission_hail_mary_nn.csv", index=False
    )
    print("Submission file saved to: \n" + LOCAL_DIR + f"submission_hail_mary_nn.csv")

    print("Training the hail mary histboost regressor.")
    model_hb = HistBoostRegressor()
    model_hb.fit(train_ensemble, train_data["DBWT"])
    lower_hb, upper_hb = model_hb.predict_intervals(test_ensemble)

    test_data[["id"]].assign(pi_lower=lower_hb, pi_upper=upper_hb).to_csv(
        LOCAL_DIR + f"submission_hail_mary_hb.csv", index=False
    )
    print("Submission file saved to: \n" + LOCAL_DIR + f"submission_hail_mary_hb.csv")
