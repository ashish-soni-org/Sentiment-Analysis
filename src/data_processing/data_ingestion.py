import os
import sys
import yaml
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.connections import s3_connection
from src.constants import AWS_ACCESS_KEY_ID_ENV_KEY, AWS_SECRET_ACCESS_KEY_ENV_KEY, S3_BUCKET, AWS_REGION

from sklearn.model_selection import train_test_split

# Stop pandas from changing data types quietly without telling us
pd.set_option("future.no_silent_downcasting", True)


def load_params(params_path: str) -> dict:
    """
    Load configuration parameters from a YAML file.

    This function reads the YAML file containing pipeline configuration
    values such as test size and returns them as a dictionary.

    Parameters
    ----------
    params_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Dictionary containing configuration parameters.
    """
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        logging.info(f"Parameters loaded from: {params_path}")
        return params

    except Exception as e:
        raise CustomException(e, sys) from e

def load_data(data_url: str) -> pd.DataFrame:
    """
    Load dataset from a CSV source.

    This method reads a CSV file from the provided path or URL
    and returns it as a pandas DataFrame.

    Parameters
    ----------
    data_url : str
        Path or URL to the CSV dataset.

    Returns
    -------
    DataFrame
        Loaded dataset.
    """
    try:
        df = pd.read_csv(data_url)
        logging.info(f"Data loaded successfully from: {data_url}")
        return df

    except Exception as e:
        raise CustomException(e, sys) from e

def preprocess_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Perform preprocessing on the raw dataset.

    This method:
    - Filters rows with valid sentiment labels
    - Converts sentiment values into numerical form

    Parameters
    ----------
    dataframe : DataFrame
        Raw input dataset.

    Returns
    -------
    DataFrame
        Preprocessed dataset.
    """
    try:
        logging.info("Starting data preprocessing")

        final_df = dataframe[dataframe["sentiment"].isin(["positive", "negative"])]
        final_df["sentiment"] = final_df["sentiment"].replace(
            {"positive": 1, "negative": 0}
        )

        logging.info("Data preprocessing completed")
        return final_df

    except Exception as e:
        raise CustomException(e, sys) from e


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """
    Save training and testing datasets to disk.

    This method creates the required directory structure
    and stores train and test datasets as CSV files.

    Parameters
    ----------
    train_data : DataFrame
        Training dataset.
    test_data : DataFrame
        Testing dataset.
    data_path : str
        Base directory path where data will be saved.
    """
    try:
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok= True)

        train_data.to_csv(
            os.path.join(raw_data_path, "train.csv"),
            index=False,
            header=True
        )
        test_data.to_csv(
            os.path.join(raw_data_path, "test.csv"),
            index=False,
            header=True
        )

        logging.info(f"Train and test data saved at: {raw_data_path}")

    except Exception as e:
        raise CustomException(e, sys) from e


def main() -> None:
    """
    Execute the data ingestion workflow.

    This function orchestrates:
    - Loading configuration parameters
    - Loading raw data
    - Preprocessing the dataset
    - Train-test splitting
    - Persisting processed data to disk
    """
    logging.info("Starting data ingestion")
    try:
        params = load_params(params_path="params.yaml")
        test_size = params["data_ingestion"]["test_size"]

        # dataframe = load_data(
        #     data_url="https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/data.csv"
        # )
        aws_access_key = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
        aws_secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)
        aws_region = os.getenv(AWS_REGION)
        bucket_name = os.getenv(S3_BUCKET)

        s3_bucket = s3_connection.S3Operations(bucket_name= bucket_name, aws_access_key= aws_access_key, aws_secret_key= aws_secret_access_key, region_name= aws_region)
        dataframe = s3_bucket.fetch_file_from_s3("IMDB.csv")

        processed_df = preprocess_data(dataframe)

        train_data, test_data = train_test_split(
            processed_df,
            test_size=test_size,
            random_state=42
        )

        save_data(
            train_data=train_data,
            test_data=test_data,
            data_path="./data"
        )

        logging.info("Data ingestion completed successfully")

    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == "__main__":

    from dotenv import load_dotenv
    load_dotenv() # To load variables from .env file
    
    main()
