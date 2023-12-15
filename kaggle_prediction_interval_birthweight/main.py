from typing import Optional

import pandas as pd
import typer

from kaggle_prediction_interval_birthweight.workflow.validation import Validator

LOCAL_DIR = "~/dev/data/kaggle-prediction-interval-birthweight/"


app = typer.Typer()


@app.command()
def create_submission(
    model_type: str = "RidgeRegressor",
    train_data_path: str = LOCAL_DIR + "train.csv",
    test_data_path: str = LOCAL_DIR + "test.csv",
    output_path: Optional[str] = None,
) -> None:
    """
    Create a submission for the competition.

    Parameters
    ----------
    model_type: str
        The type of model to use. One of RidgeRegressor, HistBoostRegressor, MissingnessNeuralNet,
        HistBoostEnsembler, or NeuralNetEnsembler
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
