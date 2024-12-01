"""
DAG for model training.
"""

from airflow.decorators import dag, task
from airflow.operators.dagrun_operator import TriggerDagRunOperator

from core.libs.train_logistic_reg_model import train_logistic_regression

default_args = {
    'owner': 'airflow',
    'retries': 1,
}

@dag(
    dag_id="model_training_dag",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description="DAG for model training."
)
def model_training_pipeline():
    """
    The pipeline for model training.
    """

    # Step 1: Trigger the data pre-processing DAG
    trigger_data_pre_processing = TriggerDagRunOperator(
        task_id="trigger_data_pre_processing",
        trigger_dag_id="pre_process_data_dag",
        wait_for_completion=True,
        reset_dag_run=True,
    )

    @task
    def train_logistic_regression_task() -> str:
        """
        Task to train logistic regression model.
        """
        train_logistic_regression()

        return "Model trained."

    info = train_logistic_regression_task()
    trigger_data_pre_processing >> info

model_training_pipeline_dag = model_training_pipeline()
