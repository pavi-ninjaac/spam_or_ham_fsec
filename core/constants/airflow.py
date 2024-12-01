"""
Constants related to airflow are here.
"""

from typing import Final

AIRFLOW_HOME_DIR: Final[str] = "/opt/airflow/"
AIRFLOW_CORE_DIR: Final[str] = "/opt/airflow/core/"
SOURCE_DATA_DIR: Final[str] = "/opt/airflow/data/source"
TARGET_DATA_DIR: Final[str] = "/opt/airflow/data/target"
TRAIN_DATA_FILE_NAME: Final[str] = "train_data.csv"
TRAIN_LABEL_FILE_NAME: Final[str] = "train_labels.csv"
TEST_DATA_FILE_NAME: Final[str] = "test_data.csv"
TEST_LABEL_FILE_NAME: Final[str] = "test_labels.csv"
X_TRAIN_FILE_NAME: Final[str] = "X_train.csv"
X_TEST_FILE_NAME: Final[str] = "X_test.csv"
Y_TRAIN_FILE_NAME: Final[str] = "y_train.csv"
Y_TEST_FILE_NAME: Final[str] = "y_test.csv"
