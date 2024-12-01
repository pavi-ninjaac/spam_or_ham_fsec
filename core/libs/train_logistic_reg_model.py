"""
Module contains the implementation of all the models.
"""

import os
from dataclasses import asdict

import dagshub
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression

from core.config.config_parser_file import LogisticRegressionConfig
from core.constants import airflow, mlflow_consts
from core.libs.utils import (
    get_classification_metrics,
    get_unique_run_name,
    log_classification_metrics,
)
from core.logger import backend_logger


def train_logistic_regression():
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

    #mlflow.set_tracking_uri(mlflow_consts.MLFLOW_TRACKING_URI)
    dagshub.init(repo_owner='pavipd495', repo_name='spam_or_ham_fsec', mlflow=True)
    mlflow.set_experiment("Spam_Or_Ham")

    param: dict = asdict(LogisticRegressionConfig())
    unique_run_name = get_unique_run_name(model_name="log_reg")

    with mlflow.start_run(run_name=unique_run_name):
        backend_logger.info(f"Shape of everything before saving the file: X_Train - {X_train.shape}, X_Test - {X_test.shape}, y_train - {y_train.shape}, y_test - {y_test.shape}")

        # Train the model.
        # param = {'C': 0.006739078892927555, 'max_iter': 483, 'solver': 'liblinear', 'penalty': 'l2'}
        model = LogisticRegression(**param)
        model.fit(X_train, y_train)

        # Log the model parameters.
        for key, value in param.items():
            mlflow.log_param(key, value)

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
        mlflow.sklearn.log_model(model, artifact_path="logistic_regression_model")

    backend_logger.info(f"Model training done for logistic regression with parameters {param}")
