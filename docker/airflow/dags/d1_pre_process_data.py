"""
The DAG which has the pipeline for pre_processing.
"""

from airflow.decorators import dag, task

from core.libs.preprocessing import (
    dimensionality_reduction_pca,
    prepare_data,
    resample_data,
    save_preprocessed_data,
    standardize_data,
)
from core.libs.train_model import train_logistic_regression

default_args = {
    'owner': 'airflow',
    'retries': 1,
}

@dag(
        dag_id="pre_process_data_dag",
        default_args=default_args,
        schedule_interval=None,
        catchup=False,
        description="DAG to preprocess data and store it in target folder."
)
def pre_processing_pipeline():
    """
    The pipeline for preprocessing data.
    """

    @task
    def prepare_data_task():
        """
        Load the data into the spark object.
        """
        X_train, X_test, y_train, y_test = prepare_data()

        # Step 2: Resample the data.
        X_train_resampled, y_train_resampled = resample_data(X_train, y_train)

        # Step 3: Standardize data.
        X_train_stand, X_test_stand = standardize_data(X_train_resampled, X_test)

        # Step 4: Dimensionality reduction using PCA.
        X_train_pca, X_test_pca = dimensionality_reduction_pca(X_train_stand, X_test_stand)

        # Step 5: Save the preprocessed data.
        save_preprocessed_data(X_train_pca, X_test_pca, y_train_resampled, y_test)

        return "done"

    @task
    def train_logistic_regression_task() -> str:
        """
        Task to train logistic regression model.
        """
        train_logistic_regression()

        return "Model trained."

    prepare_data_task() >> train_logistic_regression_task()

model_training_pipeline_dag = pre_processing_pipeline()
