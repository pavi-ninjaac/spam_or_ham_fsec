"""
Module contains the implementation of all the models.
"""

import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression

from core.config.config_parser_file import LogisticRegressionConfig
from core.constants import mlflow
from core.libs.utils import get_classification_metrics, log_classification_metrics
from core.logger import backend_logger


def train_logistic_regression():
    """
    Train the logistic regression model.
    """
    X_train = pd.read_csv("data/x_train.csv", header=None)
    X_test = pd.read_csv("data/x_text.csv", header=None)
    y_train = pd.read_csv("data/y_train.csv", header=None)
    y_test = pd.read_csv("data/y_test.csv", header=None)

    mlflow.set_tracking_uri(mlflow.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("logistic regression")

    with mlflow.start_run():
        backend_logger.info(f"Shape of everything before saving the file: X_Train - {X_train.shape}, X_Test - {X_test.shape}, y_train - {y_train.shape}, y_test - {y_test.shape}")

        # Train the model.
        param = LogisticRegressionConfig().get_config_items("param")
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
        mlflow.log_metric("confusion_matrix", metrics["confusion_matrix"])

        # Log metrics to local log file.
        log_classification_metrics(y_test=y_test, y_pred=y_pred, model_name="Logistic Regression", parameters=param)

        # Store the model artifacts.
        mlflow.sklearn.log_model(model, artifact_path="logistic_regression_model")

    backend_logger.info(f"Model training done for logistic regression with parameters {param}")
