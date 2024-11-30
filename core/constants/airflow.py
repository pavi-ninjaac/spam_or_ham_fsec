"""
Constants related to airflow are here.
"""

from typing import Final

SOURCE_DATA_DIR: Final[str] = "/opt/airflow/data/source"
TARGET_DATA_DIR: Final[str] = "/opt/airflow/data/target"
TRAIN_DATA_FILE_NAME: Final[str] = "train_data.csv"
TRAIN_LABEL_FILE_NAME: Final[str] = "train_labels.csv"
TEST_DATA_FILE_NAME: Final[str] = "test_data.csv"
TEST_LABEL_FILE_NAME: Final[str] = "test_labels.csv"
