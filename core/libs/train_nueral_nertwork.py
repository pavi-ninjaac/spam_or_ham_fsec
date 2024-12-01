"""
Module contains the implementation of all the models.
"""

import os
from dataclasses import asdict

import mlflow
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

from core.config.config_parser_file import NueralNetwork
from core.constants import airflow, mlflow_consts
from core.libs.utils import (
    get_classification_metrics,
    get_unique_run_name,
    log_classification_metrics,
)
from core.logger import backend_logger


def train_nueral_network():
    """
    Train the logistic regression model.
    """
    x_train_file: str = os.path.join(airflow.TARGET_DATA_DIR, airflow.X_TRAIN_FILE_NAME)
    y_train_file: str = os.path.join(airflow.TARGET_DATA_DIR, airflow.Y_TRAIN_FILE_NAME)
    x_test_file: str = os.path.join(airflow.TARGET_DATA_DIR, airflow.X_TEST_FILE_NAME)
    y_test_file: str = os.path.join(airflow.TARGET_DATA_DIR, airflow.Y_TEST_FILE_NAME)

    X_train = pd.read_csv(x_train_file, header=None)
    X_test = pd.read_csv(x_test_file, header=None)
    y_train = pd.read_csv(y_train_file, header=None)
    y_test = pd.read_csv(y_test_file, header=None)

    mlflow.set_tracking_uri(mlflow_consts.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("logistic regression")

    param: dict = asdict(())
    unique_run_name = get_unique_run_name(model_name="nueral_network")

    with mlflow.start_run(run_name=unique_run_name):
        backend_logger.info(f"Shape of everything before saving the file: X_Train - {X_train.shape}, X_Test - {X_test.shape}, y_train - {y_train.shape}, y_test - {y_test.shape}")

        layer1_units = 0
        layer2_units = 0
        dropout_rate = 0
        learning_rate = 0

        model = Sequential()
        model.add(Dense(layer1_units, activation='relu', input_dim=X_train.shape[1]))
        model.add(Dropout(dropout_rate))
        model.add(Dense(layer2_units, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(learning_rate=learning_rate),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        # Get the predictions.
        y_pred = model.predict(X_test)

        # Log metrics to mlflow.
        metrics = get_classification_metrics(y_test=y_test, y_pred=y_pred)
        mlflow.log_metric("accuracy", metrics["accuracy"],)
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1_score", metrics["f1_score"])
        #mlflow.log_metric("confusion_matrix", metrics["confusion_matrix"])

        # Log metrics to local log file.
        log_classification_metrics(y_test=y_test, y_pred=y_pred, model_name="Logistic Regression", parameters=param)

        # Store the model artifacts.
        mlflow.sklearn.log_model(model, artifact_path="nueral_network_model")

    backend_logger.info(f"Model training done for nueral network with parameters {param}")
