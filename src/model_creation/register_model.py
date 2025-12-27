import os
import sys
import json
import mlflow
from mlflow.tracking import MlflowClient
from from_root import from_root

from src.logger import logging
from src.constants import DAGSHUB_TOKEN_ENV_KEY
from src.exception import CustomException


def setup_mlflow_tracking() -> None:
    """Configure MLflow tracking for model registry."""
    try:
        dagshub_token = os.getenv(DAGSHUB_TOKEN_ENV_KEY)
        if not dagshub_token:
            raise EnvironmentError(f"{DAGSHUB_TOKEN_ENV_KEY} environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "ashishsoni295work"
        repo_name = "Sentiment-Analysis"

        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        logging.info("MLflow tracking configured with DagsHub")

    except Exception as e:
        raise CustomException(e, sys) from e


def load_model_info(file_path: str) -> dict:
    """Load MLflow run metadata."""
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)
        logging.info(f"Model info loaded from: {file_path}")
        return model_info
    except Exception as e:
        raise CustomException(e, sys) from e


def register_model(model_name: str, model_info: dict) -> None:
    """
    Register and transition model to Staging using MlflowClient.
    
    This method uses the lower-level MlflowClient to avoid 400 errors
    associated with the high-level register_model wrapper when a model
    name already exists.
    """
    try:
        client = MlflowClient()
        run_id = model_info['run_id']
        model_path = model_info['model_path']
        
        # 1. Ensure the Registered Model exists
        try:
            client.create_registered_model(model_name)
            logging.info(f"Created new registered model: {model_name}")
        except Exception:
            logging.info(f"Registered model '{model_name}' already exists. Proceeding to create version.")

        # 2. Create the Model Version
        # The source URI must be exactly what MLflow expects
        source_uri = f"runs:/{run_id}/{model_path}"
        
        logging.info(f"Creating model version from source: {source_uri}")
        
        model_version = client.create_model_version(
            name=model_name,
            source=source_uri,
            run_id=run_id
        )

        # 3. Transition to Staging
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logging.info(f"Model '{model_name}' version {model_version.version} registered and transitioned to Staging")

    except Exception as e:
        raise CustomException(e, sys) from e


def main() -> None:
    """Execute model registration pipeline."""
    logging.info("Starting model registration")
    try:
        setup_mlflow_tracking()

        model_info_path = os.path.join(from_root(), "reports", "experiment_info.json")
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
    load_dotenv()
    main()