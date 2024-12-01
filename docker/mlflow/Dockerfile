# Use the official Python image
FROM python:3.10-slim

# Set a working directory
WORKDIR /app

# Install system dependencies (including any optional ones for MLflow integration)
RUN apt-get update && apt-get install -y \
    build-essential \
    default-libmysqlclient-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow and any additional Python dependencies (e.g., for cloud storage and DB support)
RUN pip install --no-cache-dir mlflow[extras] boto3 pymysql psycopg2-binary

# Create a directory for MLflow logs and artifacts
RUN mkdir -p /app/mlruns
RUN chmod -R 777 /app


ENV MLFLOW_HOME=/app/mlflow

ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Expose the default MLflow port
EXPOSE 5000

# Set the entrypoint command to start the MLflow server
# CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "file:/app/mlruns", "--host", "0.0.0.0"]
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]
