"""
The setting related to logs are present here.
"""

import logging
import os

from constants import airflow

logging_str: str = "[%(asctime)s - %(levelname)s - %(module)s.py - %(name)s]: %(message)s"

# Create the log directory.
log_dir: str = os.path.join(airflow.AIRFLOW_CORE_DIR, "logs")
backend_log_filename: str = "back_end.log"
backend_log_filepath: str = os.path.join(log_dir, backend_log_filename)

os.makedirs(log_dir, exist_ok=True)

# Set the logging config.
logging.basicConfig(
    level=logging.DEBUG,
    format=logging_str,
    # handlers=[
    #     logging.FileHandler(log_filepath),
    #     logging.StreamHandler(sys.stdout)
    # ]
)

# Get the logger.
backend_file_handler = logging.FileHandler(backend_log_filepath)

# add the formettor for the handler,
backend_file_handler.setFormatter(logging.Formatter(logging_str))

backend_logger = logging.getLogger("backend")
frontend_logger = logging.getLogger("frontend")


# add the handlers.
backend_logger.addHandler(backend_file_handler)
