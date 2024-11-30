"""
This module contains the code for data pre-processing.
"""
import os

import pandas as pd

from core.constants import airflow


def pca():
    """

    """
    # Load the data from the source dir.
    data: pd.DataFrame = pd.read_csv(os.path.join(airflow.SOURCE_DATA_DIR, airflow.TRAIN_DATA_FILE_NAME))

    #