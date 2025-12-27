import os
import sys
import yaml
import pandas as pd
from from_root import from_root
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src.connections.s3_connection import S3Operations
from src.constants import (
    AWS_ACCESS_KEY_ID_ENV_KEY, 
    AWS_SECRET_ACCESS_KEY_ENV_KEY, 
    S3_BUCKET_ENV_KEY, 
    AWS_REGION_ENV_KEY
)

# Stop pandas from changing data types quietly
pd.set_option("future.no_silent_downcasting", True)


def load_params(params_path: str) -> dict:
    """Load configuration parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from: {params_path}")
        return params
    except Exception as e:
        raise CustomException(e, sys) from e


def preprocess_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Filter rows and convert sentiment labels to numerical form."""
    try:
        logging.info("Starting data ingestion preprocessing")
        
        # Filter only positive and negative reviews
        final_df = dataframe[dataframe["sentiment"].isin(["positive", "negative"])].copy()
        
        # Map to binary
        final_df["sentiment"] = final_df["sentiment"].replace(
            {"positive": 1, "negative": 0}
        )
        
        logging.info("Data ingestion preprocessing completed")
        return final_df
    except Exception as e:
        raise CustomException(e, sys) from e


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, base_path: str) -> None:
    """Save training and testing datasets to disk."""
    try:
        raw_data_path = os.path.join(base_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)

        train_path = os.path.join(raw_data_path, "train.csv")
        test_path = os.path.join(raw_data_path, "test.csv")

        train_data.to_csv(train_path, index=False, header=True)
        test_data.to_csv(test_path, index=False, header=True)

        logging.info(f"Train and test data saved at: {raw_data_path}")
    except Exception as e:
        raise CustomException(e, sys) from e


def main() -> None:
    """Execute the data ingestion workflow."""
    logging.info("Starting data ingestion pipeline")
    try:
        # Load params using absolute path safe for DVC
        params = load_params(params_path=os.path.join(from_root(), "params.yaml"))
        test_size = params["data_ingestion"]["test_size"]

        # Fetch Credentials
        aws_access_key = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
        aws_secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)
        aws_region = os.getenv(AWS_REGION_ENV_KEY)
        bucket_name = os.getenv(S3_BUCKET_ENV_KEY)

        if not all([aws_access_key, aws_secret_access_key, bucket_name]):
            raise EnvironmentError("AWS credentials or Bucket Name missing in environment variables.")

        # S3 Operation
        s3_ops = S3Operations(
            bucket_name=bucket_name, 
            aws_access_key=aws_access_key, 
            aws_secret_key=aws_secret_access_key, 
            region_name=aws_region
        )
        
        # Fetch Data
        dataframe = s3_ops.fetch_file_from_s3("IMDB.csv")

        # Process
        processed_df = preprocess_data(dataframe)

        # Split
        train_data, test_data = train_test_split(
            processed_df,
            test_size=test_size,
            random_state=42
        )

        # Save
        save_data(
            train_data=train_data,
            test_data=test_data,
            base_path=os.path.join(from_root(), "data")
        )

        logging.info("Data ingestion pipeline completed successfully")

    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()