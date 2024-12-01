"""
Contains the preprocessing functions for the pipeline.
"""
import os

import numpy as np
import pandas as pd
from constants import airflow
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from logger import backend_logger
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from core.config.config_parser_file import PreprocessConfig
from core.logger import backend_logger


def prepare_data() -> tuple[pd.DataFrame]:
    """
    Read the data from the source folder and split it.
    """
    source_data_path: str = os.path.join(airflow.SOURCE_DATA_DIR, airflow.TRAIN_DATA_FILE_NAME)
    source_target_path: str = os.path.join(airflow.SOURCE_DATA_DIR, airflow.TRAIN_LABEL_FILE_NAME)
    try:
        data = pd.read_csv(source_data_path, header=None)
        target = pd.read_csv(source_target_path, header=None)
    except FileNotFoundError as e:
        backend_logger.error(f"File not found {str(e)}")
        return
    # Split the data into train and test.
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=42, stratify=target)

    backend_logger.info(f"The data is spilited. the shape of it are X-train:{X_train.shape}, X_test: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def resample_data(X_train: pd.DataFrame, y_train: pd.DataFrame,
                  ):
    """
    Re-sample the data, oversample and undersample if given.

    :param X_train: Dataframe of x train.
    :param y_train: Dataframe of y train.
    """
    config = PreprocessConfig().get_config_items("param")
    oversample_sampling_strategy: float = float(config["oversample_sampling_strategy"])
    undersample_sampling_strategy: float = float(config["undersample_sampling_strategy"])
    do_oversample: bool= bool(config["do_oversample"])
    do_undersample: bool= bool(config["do_undersample"])

    backend_logger.info("Sampling is enabled.")
    if do_oversample:
        backend_logger.info(f"Oversmapling is happening with the sampling strategy of: {oversample_sampling_strategy}")
        backend_logger.info(f"The shape of X_train before oversampling is {X_train.shape}")
        backend_logger.info(f"The class distribution: {(y_train.iloc[:,0].value_counts() / y_train.shape[0]) * 100}")

        smote = SMOTE(sampling_strategy=oversample_sampling_strategy, k_neighbors=7, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        backend_logger.info(f"The shape of X_train after oversampling is: {X_train.shape}")
        backend_logger.info(f"The class distribution: {(y_train.iloc[:,0].value_counts() / y_train.shape[0]) * 100}")

    # Do undersampling if mentioned.
    if do_undersample:
        backend_logger.info(f"undersampling is happening with the sampling strategy of: {undersample_sampling_strategy}")
        backend_logger.info(f"The shape of X_train before undersampling is {X_train.shape}")
        backend_logger.info(f"The class distribution: {(y_train.iloc[:,0].value_counts() / y_train.shape[0]) * 100}")

        undersampler = RandomUnderSampler(sampling_strategy=undersample_sampling_strategy, random_state=42)
        X_train, y_train = undersampler.fit_resample(X_train, y_train)

        backend_logger.info(f"The shape of X_train after undersampling is: {X_train.shape}")
        backend_logger.info(f"The class distribution: {(y_train.iloc[:,0].value_counts() / y_train.shape[0]) * 100}")

    return X_train, y_train


def standardize_data(X_train: pd.DataFrame, X_test:pd.DataFrame):
    """
    Do StandardScaler operation on the training and test sets.

    :param X_train: Dataframe of x train.
    :param X_test: Dataframe of x test.
    """
    scaler = StandardScaler()
    backend_logger.info("Standard scaling is happening...")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    backend_logger.info("Standard scaling is completed successfully.")

    return X_train_scaled, X_test_scaled


def dimensionality_reduction_pca(X_train: pd.DataFrame, X_test:pd.DataFrame):
    """
    Compute the dimensionality reduction for the given X_train and X_test vectors.

    :param X_train: Dataframe of x train.
    :param X_test: Dataframe of x test.
    """
    config = PreprocessConfig().get_config_items("param")
    n_components = int(config["n_components"])
    pca = PCA(n_components=n_components)

    backend_logger.info(f"PCA is happening with the n_components of {n_components}...")
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    backend_logger.info("PCA is completed successfully.")

    backend_logger.info(f"Shape of everything after PCA: X_Train - {X_train_pca.shape}, X_Test - {X_test_pca.shape}")

    return X_train_pca, X_test_pca


def save_preprocessed_data(X_train_pca: np.ndarray, X_test_pca: np.ndarray, y_train: pd.DataFrame, y_test: pd.DataFrame):
    """
    Save the preprocessed data into the target folder.

    :param X_train_pca: Dataframe of x train after PCA.
    :param X_test_pca: Dataframe of x test after PCA.
    :param y_train: Dataframe of y train.
    :param y_test: Dataframe of y test.
    """
    backend_logger.info(f"Shape of everything before saving the file: X_Train - {X_train_pca.shape}, X_Test - {X_test_pca.shape}, y_train - {y_train.shape}, y_test - {y_test.shape}")

    x_train_file: str = os.path.join(airflow.TARGET_DATA_DIR, airflow.X_TRAIN_FILE_NAME)
    y_train_file: str = os.path.join(airflow.TARGET_DATA_DIR, airflow.Y_TRAIN_FILE_NAME)

    x_test_file: str = os.path.join(airflow.TARGET_DATA_DIR, airflow.X_TEST_FILE_NAME)
    y_test_file: str = os.path.join(airflow.TARGET_DATA_DIR, airflow.Y_TEST_FILE_NAME)

    np.savetxt(x_train_file, X_train_pca, delimiter=',')
    np.savetxt(x_test_file, X_test_pca, delimiter=',')
    np.savetxt(y_train_file, y_train, delimiter=',', fmt='%d')
    np.savetxt(y_test_file, y_test, delimiter=',', fmt='%d')

    backend_logger.info("The data is saved successfully to the target folder. :)")


def replace_target_class_name(y_train, y_test):
    """
    replace -1 to 1 and 1 to 0.
    """
    y_train.replace(1, 0, inplace=True)
    y_train.replace(-1, 1, inplace=True)


    y_test.replace(1, 0, inplace=True)
    y_test.replace(-1, 1, inplace=True)

    return y_train, y_test
