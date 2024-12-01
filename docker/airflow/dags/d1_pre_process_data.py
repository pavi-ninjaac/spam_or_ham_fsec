"""
The DAG which has the pipeline for pre_processing.
"""
import os

from airflow.decorators import dag, task

from core.constants.airflow import (
    SOURCE_DATA_DIR,
    TARGET_DATA_DIR,
    TRAIN_DATA_FILE_NAME,
    X_TRAIN_FILE_NAME,
)
from core.libs.preprocessing import (
    dimensionality_reduction_pca,
    prepare_data,
    replace_target_class_name,
    resample_data,
    save_preprocessed_data,
    standardize_data,
)
from core.libs.train_logistic_reg_model import train_logistic_regression

#from core.libs.train_nueral_nertwork import train_nueral_network
from core.libs.train_random_forest import train_random_forest
from core.libs.train_svc_model import train_svc

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
    def check_dataset():
        """
        Check if the dataset exists.
        """
        # Check if the dataset exists in the source folder.
        # If not, raise an exception.
        if not os.path.exists(os.path.join(SOURCE_DATA_DIR, TRAIN_DATA_FILE_NAME)):
            raise FileNotFoundError("The source data file is not found.")
        return "Move on"

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

        # Step 5: replace the target values.
        y_train_fi, y_test_fi = replace_target_class_name(y_train_resampled, y_test)

        # Step 6: Save the preprocessed data.
        save_preprocessed_data(X_train_pca, X_test_pca, y_train_fi, y_test_fi)

        return "done"

    @task
    def check_preprocessed_dataset():
        """
        Check if the dataset exists.
        """
        # Check if the dataset exists in the source folder.
        # If not, raise an exception.
        if not os.path.exists(os.path.join(TARGET_DATA_DIR, X_TRAIN_FILE_NAME)):
            raise FileNotFoundError("The preprocessed data files are not found.")
        return "Move on"

    @task
    def train_logistic_regression_task() -> str:
        """
        Task to train logistic regression model.
        """
        train_logistic_regression()

        return "Model trained."

    @task
    def train_svm_classification_task() -> str:
        """
        Task to train SVM classification model.
        """
        train_svc()

        return "SVM Model trained."

    @task
    def train_random_forest_classification_task() -> str:
        """
        Task to train Random Forest classification model.
        """
        train_random_forest()

        return "Random Forest Model trained."


    info = check_dataset()
    # info >> prepare_data_task() >> check_preprocessed_dataset()
    info >> prepare_data_task() >> check_preprocessed_dataset() >> [train_logistic_regression_task(), train_svm_classification_task(),
                                                                    train_random_forest_classification_task()]


model_training_pipeline_dag = pre_processing_pipeline()
