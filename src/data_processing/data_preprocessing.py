import os
import sys
import re
import string
import pandas as pd
from pandas import DataFrame
from from_root import from_root

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.logger import logging
from src.exception import CustomException


class TextPreprocessor:
    """
    Handles text preprocessing operations for NLP pipelines.
    """

    def __init__(self):
        """Initialize text preprocessing dependencies."""
        try:
            # Ensure resources are downloaded
            nltk.download("wordnet", quiet=True)
            nltk.download("stopwords", quiet=True)
            nltk.download('omw-1.4', quiet=True)

            self.lemmatizer = WordNetLemmatizer()
            
            # CRITICAL: Exclude negation words from the stopword list
            # The model MUST see 'not', 'no', 'never' to understand sentiment.
            all_stopwords = set(stopwords.words("english"))
            negation_words = {"not", "no", "nor", "neither", "never", "none", "n't"}
            self.stop_words = all_stopwords - negation_words

            logging.info("TextPreprocessor initialized (Negation words preserved)")

        except Exception as e:
            raise CustomException(e, sys) from e

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize a single text entry."""
        try:
            # Lowercase first
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r"https?://\S+|www\.\S+", "", text)
            
            # Remove HTML tags
            text = re.sub(r'<.*?>', '', text)
            
            # Handle standard contractions before punctuation removal
            text = re.sub(r"n't", " not", text)
            text = re.sub(r"'re", " are", text)
            text = re.sub(r"'s", " is", text)
            text = re.sub(r"'d", " would", text)
            text = re.sub(r"'ll", " will", text)
            text = re.sub(r"'ve", " have", text)
            text = re.sub(r"'m", " am", text)

            # Keep only characters and spaces (remove digits/punctuation)
            # We treat punctuation as space to avoid merging words
            text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
            text = re.sub(r"\d+", "", text)
            
            # Normalize whitespace
            text = re.sub(r"\s+", " ", text).strip()

            # Remove Stopwords (Preserving Negations)
            text = " ".join(
                word for word in text.split() if word not in self.stop_words
            )

            # Lemmatization
            text = " ".join(
                self.lemmatizer.lemmatize(word) for word in text.split()
            )

            return text

        except Exception as e:
            raise CustomException(e, sys) from e

    def preprocess_dataframe(self, dataframe: DataFrame, text_column: str) -> DataFrame:
        """Apply text preprocessing to a specified column."""
        try:
            logging.info(f"Starting text preprocessing on column: {text_column}")

            dataframe[text_column] = dataframe[text_column].astype(str).apply(self._preprocess_text)
            
            # Drop rows where text might have become empty after cleaning
            dataframe = dataframe[dataframe[text_column].str.strip() != ""]

            logging.info("Text preprocessing on column completed successfully")
            return dataframe

        except Exception as e:
            raise CustomException(e, sys) from e


def main() -> None:
    """Execute the text preprocessing workflow."""
    logging.info("Starting data preprocessing pipeline")
    try:
        base_data_path = os.path.join(from_root(), "data")
        
        train_path = os.path.join(base_data_path, "raw", "train.csv")
        test_path = os.path.join(base_data_path, "raw", "test.csv")
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        logging.info("Raw train and test data loaded successfully")

        text_preprocessor = TextPreprocessor()

        train_processed = text_preprocessor.preprocess_dataframe(dataframe=train_data, text_column="review")
        test_processed = text_preprocessor.preprocess_dataframe(dataframe=test_data, text_column="review")

        output_path = os.path.join(base_data_path, "interim")
        os.makedirs(output_path, exist_ok=True)

        train_processed.to_csv(os.path.join(output_path, "train_processed.csv"), index=False, header=True)
        test_processed.to_csv(os.path.join(output_path, "test_processed.csv"), index=False, header=True)

        logging.info(f"Processed data saved at: {output_path}")

    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    main()