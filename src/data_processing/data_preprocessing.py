import os
import sys
import re
import string

import numpy as np
import pandas as pd
from pandas import DataFrame

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.logger import logging
from src.exception import CustomException


class TextPreprocessor:
    """
    Handles text preprocessing operations for NLP pipelines.

    This class is responsible for cleaning, normalizing, and
    lemmatizing text data prior to feature engineering or
    model training.
    """

    def __init__(self):
        """
        Initialize text preprocessing dependencies.

        This method ensures required NLTK resources are available
        before executing preprocessing steps.
        """
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("stopwords", quiet=True)

            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words("english"))

            logging.info("TextPreprocessor initialized successfully")

        except Exception as e:
            raise CustomException(e, sys) from e

    def _preprocess_text(self, text: str) -> str:
        """
        Apply text cleaning and normalization to a single text entry.

        This method:
        - Removes URLs and numbers
        - Converts text to lowercase
        - Removes punctuation and extra spaces
        - Removes stopwords
        - Applies lemmatization

        Parameters
        ----------
        text : str
            Raw text string.

        Returns
        -------
        str
            Cleaned and normalized text.
        """
        try:
            text = re.sub(r"https?://\S+|www\.\S+", "", text)
            text = "".join(char for char in text if not char.isdigit())
            text = text.lower()

            text = re.sub(
                "[%s]" % re.escape(string.punctuation),
                " ",
                text
            )
            text = text.replace("؛", "")
            text = re.sub(r"\s+", " ", text).strip()

            text = " ".join(
                word for word in text.split() if word not in self.stop_words
            )

            text = " ".join(
                self.lemmatizer.lemmatize(word) for word in text.split()
            )

            return text

        except Exception as e:
            raise CustomException(e, sys) from e

    def preprocess_dataframe(self, dataframe: DataFrame, text_column: str) -> DataFrame:
        """
        Apply text preprocessing to a specified column in a DataFrame.

        Parameters
        ----------
        dataframe : DataFrame
            Input dataset containing raw text.
        text_column : str
            Name of the column containing text data.

        Returns
        -------
        DataFrame
            DataFrame with preprocessed text.
        """
        try:
            logging.info(
                f"Starting text preprocessing on column: {text_column}"
            )

            dataframe[text_column] = dataframe[text_column].apply(self._preprocess_text)

            dataframe = dataframe.dropna(subset=[text_column])

            logging.info("Text preprocessing on column completed successfully")
            return dataframe

        except Exception as e:
            raise CustomException(e, sys) from e


def main() -> None:
    """
    Execute the text preprocessing workflow.

    This function:
    - Loads raw training and testing datasets
    - Applies text preprocessing
    - Saves processed data to the interim directory
    """
    logging.info("Starting text preprocessing")
    try:
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")

        logging.info("Raw train and test data loaded successfully")

        text_preprocessor = TextPreprocessor()

        train_processed_data = text_preprocessor.preprocess_dataframe(dataframe=train_data, text_column="review")
        test_processed_data = text_preprocessor.preprocess_dataframe(dataframe=test_data, text_column="review")

        output_path = os.path.join("./data", "interim")
        os.makedirs(output_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(output_path, "train_processed.csv"), index=False, header=True)
        test_processed_data.to_csv(os.path.join(output_path, "test_processed.csv"), index=False, header=True)

        logging.info(f"Processed data saved at: {output_path}")

    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    main()
