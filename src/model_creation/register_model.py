import os
import sys
import json
import warnings

import mlflow
import dagshub

from src.logger import logging
from src.constants import DAGSHUB_TOKEN
from src.exception import CustomException


def setup_mlflow_tracking() -> None:
    """
    Configure MLflow tracking with DagsHub.

    This method validates required environment variables and
    sets the MLflow tracking URI for model registration.
    """
    try:
        dagshub_token = os.getenv(DAGSHUB_TOKEN)
        if not dagshub_token:
            raise EnvironmentError(
                "DAGSHUB_TOKEN environment variable is not set"
            )

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "ashishsoni295work"
        repo_name = "Sentiment-Analysis"

        mlflow.set_tracking_uri(
            f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
        )

        logging.info("MLflow tracking configured with DagsHub")

    except Exception as e:
        raise CustomException(e, sys) from e


def load_model_info(file_path: str) -> dict:
    """
    Load MLflow run metadata from disk.

    Parameters
    ----------
    file_path : str
        Path to the JSON file containing run information.

    Returns
    -------
    dict
        Dictionary containing MLflow run ID and model path.
    """
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)

        logging.info(f"Model info loaded from: {file_path}")
        return model_info

    except Exception as e:
        raise CustomException(e, sys) from e


def register_model(
    model_name: str,
    model_info: dict
) -> None:
    """
    Register a trained model in the MLflow Model Registry.

    This method:
    - Registers the model using MLflow run artifacts
    - Transitions the model version to the Staging stage

    Parameters
    ----------
    model_name : str
        Name under which the model will be registered.
    model_info : dict
        Dictionary containing run ID and model artifact path.
    """
    try:
        model_uri = (
            f"runs:/{model_info['run_id']}/"
            f"{model_info['model_path']}"
        )

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logging.info(
            f"Model '{model_name}' version "
            f"{model_version.version} registered and "
            f"transitioned to Staging"
        )

    except Exception as e:
        raise CustomException(e, sys) from e


def main() -> None:
    """
    Execute the model registration pipeline.

    This function orchestrates:
    - MLflow and DagsHub setup
    - Loading model run metadata
    - Registering the trained model
    """
    logging.info("Starting model registration")
    try:

        setup_mlflow_tracking()

        model_info_path = "./reports/experiment_info.json"
        model_info = load_model_info(model_info_path)

        model_name = "sentiment_analysis_model"
        register_model(
            model_name=model_name,
            model_info=model_info
        )

        logging.info("Model registration completed successfully")

    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == "__main__":

    from dotenv import load_dotenv
    load_dotenv() # To load variables from .env file
    
    main()