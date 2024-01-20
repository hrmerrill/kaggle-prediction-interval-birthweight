from typing import Optional

import numpy as np
import pandas as pd
import typer
from tensorflow.random import set_seed

from kaggle_prediction_interval_birthweight.model.hist_gradient_boosting import HistBoostRegressor
from kaggle_prediction_interval_birthweight.model.neural_network import (
    SOFTPLUS_SCALE,
    MissingnessNeuralNetRegressor,
)
from kaggle_prediction_interval_birthweight.model.wildwood import WildWoodRegressor
from kaggle_prediction_interval_birthweight.utils.utils import np_softplus_inv
from kaggle_prediction_interval_birthweight.workflow.validation import Validator

LOCAL_DIR = "~/dev/data/kaggle-prediction-interval-birthweight/"

# for reproducibility
np.random.seed(1)
set_seed(1)


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
        The type of model to use. One of RidgeRegressor, HistBoostRegressor, WildWoodRegressor,
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

    # save predictions on training sets too, so they can be ensembled later
    fitted_path = LOCAL_DIR + f"fitted_{model_type}.csv"
    lower_train, upper_train = validator.predict_intervals(train_data)
    fitted_data = train_data[["id"]].copy()
    fitted_data[f"lower_{model_type}"] = lower_train
    fitted_data[f"upper_{model_type}"] = upper_train
    fitted_data.to_csv(fitted_path, index=False)
    print("Fitted file saved to: \n" + fitted_path)


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
        "WildWoodRegressor",
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

        train_ensemble = train_ensemble + [
            lower_train / SOFTPLUS_SCALE,
            upper_train / SOFTPLUS_SCALE,
        ]
        test_ensemble = test_ensemble + [lower_test / SOFTPLUS_SCALE, upper_test / SOFTPLUS_SCALE]

        print(f"{model_type} complete.")

    train_ensemble = np.hstack([x.reshape((-1, 1)) for x in train_ensemble])
    test_ensemble = np.hstack([x.reshape((-1, 1)) for x in test_ensemble])
    train_y = np_softplus_inv(train_data["DBWT"] / SOFTPLUS_SCALE)

    print("Training the hail mary wildwood regressor.")
    model_ww = WildWoodRegressor()
    model_ww.fit(train_ensemble, train_data["DBWT"])
    lower_ww, upper_ww = model_ww.predict_intervals(test_ensemble)

    test_data[["id"]].assign(pi_lower=lower_ww, pi_upper=upper_ww).to_csv(
        LOCAL_DIR + "submission_hail_mary_ww.csv", index=False
    )
    print("Submission file saved to: \n" + LOCAL_DIR + "submission_hail_mary_ww.csv")

    print("Training the hail mary histboost regressor.")
    model_hb = HistBoostRegressor()
    model_hb.fit(train_ensemble, train_data["DBWT"])
    lower_hb, upper_hb = model_hb.predict_intervals(test_ensemble)

    test_data[["id"]].assign(pi_lower=lower_hb, pi_upper=upper_hb).to_csv(
        LOCAL_DIR + "submission_hail_mary_hb.csv", index=False
    )
    print("Submission file saved to: \n" + LOCAL_DIR + "submission_hail_mary_hb.csv")

    print("Training the hail mary neural network.")
    model_nn = MissingnessNeuralNetRegressor(bayesian=True, fit_tail=False)
    model_nn.fit(train_ensemble, train_y)
    lower_nn, upper_nn = model_nn.predict_intervals(test_ensemble, alpha=0.9, n_samples=2000)

    test_data[["id"]].assign(pi_lower=lower_nn, pi_upper=upper_nn).to_csv(
        LOCAL_DIR + "submission_hail_mary_nn.csv", index=False
    )
    print("Submission file saved to: \n" + LOCAL_DIR + "submission_hail_mary_nn.csv")
