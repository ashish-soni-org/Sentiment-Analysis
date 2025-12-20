import sys
from io import StringIO

import boto3
import pandas as pd

from src.logger import logging
from src.exception import CustomException


class S3Operations:
    """
    Handles interaction with AWS S3 for data retrieval.

    This class is responsible for connecting to an S3 bucket and
    fetching CSV files as pandas DataFrames for downstream
    data ingestion and preprocessing.
    """

    def __init__(self, bucket_name: str, aws_access_key: str, aws_secret_key: str, region_name: str):
        """
        Initialize the S3Operations component with AWS credentials.

        Parameters
        ----------
        bucket_name : str
            Name of the S3 bucket.
        aws_access_key : str
            AWS access key ID.
        aws_secret_key : str
            AWS secret access key.
        region_name : str, optional
            AWS region name, by default "us-east-1".
        """
        try:
            self.bucket_name = bucket_name
            self.s3_client = boto3.client(
                service_name="s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region_name
            )

            logging.info(
                f"S3Operations initialized for bucket: {self.bucket_name}"
            )

        except Exception as e:
            raise CustomException(e, sys) from e

    def fetch_file_from_s3(self, file_key: str) -> pd.DataFrame:
        """
        Fetch a CSV file from the configured S3 bucket.

        This method:
        - Downloads the object from S3
        - Reads its content into memory
        - Converts it into a pandas DataFrame

        Parameters
        ----------
        file_key : str
            Path of the file inside the S3 bucket (e.g., 'data/data.csv').

        Returns
        -------
        DataFrame
            DataFrame containing the loaded CSV data.
        """
        try:
            logging.info(f"Fetching file '{file_key}' from S3 bucket '{self.bucket_name}'")

            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=file_key
            )

            dataframe = pd.read_csv(
                StringIO(response["Body"].read().decode("utf-8"))
            )

            logging.info(
                f"Successfully fetched '{file_key}' "
                f"with shape: {dataframe.shape}"
            )

            return dataframe

        except Exception as e:
            raise CustomException(e, sys) from e
