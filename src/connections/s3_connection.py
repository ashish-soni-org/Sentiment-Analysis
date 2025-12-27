import sys
import boto3
import pandas as pd
from io import StringIO
from typing import Optional

from src.logger import logging
from src.exception import CustomException

class S3Operations:
    """
    Handles interaction with AWS S3 for data retrieval.
    """

    def __init__(self, bucket_name: str, aws_access_key: str, aws_secret_key: str, region_name: str = "us-east-1"):
        """
        Initialize the S3Operations component with AWS credentials.
        """
        try:
            self.bucket_name = bucket_name
            self.s3_client = boto3.client(
                service_name="s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region_name
            )

            logging.info(f"S3Operations initialized for bucket: {self.bucket_name}")

        except Exception as e:
            raise CustomException(e, sys) from e

    def fetch_file_from_s3(self, file_key: str) -> pd.DataFrame:
        """
        Fetch a CSV file from the configured S3 bucket.
        """
        try:
            logging.info(f"Fetching file '{file_key}' from S3 bucket '{self.bucket_name}'")

            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=file_key
            )

            content = response["Body"].read().decode("utf-8")
            dataframe = pd.read_csv(StringIO(content))

            logging.info(f"Successfully fetched '{file_key}' with shape: {dataframe.shape}")

            return dataframe

        except Exception as e:
            raise CustomException(e, sys) from e