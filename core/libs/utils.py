"""
The module contains the untility functions.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from core.logger import backend_logger


def log_classification_metrics(y_test: pd.DataFrame, y_pred: pd.DataFrame,
                           model_name: str, parameters: dict[str, str]):
    """
    Log the classification report with the model name and parameters.
    """
    backend_logger.info("="*50)
    backend_logger.info(f"Model Name: {model_name} and Parameters: {parameters}")
    backend_logger.info("="*50)
    backend_logger.info(f"{'='*50}")
    backend_logger.info(f"The accuracy score: {accuracy_score(y_test, y_pred)}")
    backend_logger.info(f"Precision: {precision_score(y_test, y_pred)}")
    backend_logger.info(f"Recall: {recall_score(y_test, y_pred)}")
    backend_logger.info(f"F1 score: {f1_score(y_test, y_pred)}")
    backend_logger.info(f"{'='*50}")
    backend_logger.info(f"{classification_report(y_test, y_pred)}")
    backend_logger.info("\n")
    backend_logger.info(f"{'='*50}")
    backend_logger.info("confusion Matrix")
    backend_logger.info(f"{confusion_matrix(y_test, y_pred)}")
    backend_logger.info(f"{'='*50}")


def get_classification_metrics(y_test: pd.DataFrame, y_pred: pd.DataFrame) -> dict[str, str]:
    """
    Returns the classification report as a dictionary.
    """
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
