"""
Dag file for the collect data task.
"""

import mlflow
import sklearn
from airflow.decorators import dag, task

default_args = {
    'owner': 'airflow',
    'retries': 1,
}


@dag(
        dag_id="data_collection_dag",
        default_args=default_args,
        schedule_interval=None,  # Run on demand
        catchup=False,
        description="DAG to collect data and store it in a target location"
)
def data_collection_pipeline():
    """
    The pipeline for collecting the data.
    """

    @task
    def collect_data():
        """
        Collect file task.

        :param file_name: The file name.
        """
        print("imported sklearn")
        print("Collecting the data.")

    collect_data()

data_collection_pipeline_dag = data_collection_pipeline()
