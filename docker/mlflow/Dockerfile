# Use the official Python image
FROM python:3.10-slim

# Set a working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    default-libmysqlclient-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow and any additional Python dependencies
RUN pip install --no-cache-dir mlflow[extras] boto3 pymysql psycopg2-binary

# Expose the default MLflow port
EXPOSE 5000

# Set the entrypoint command to start the MLflow server
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "file:/app/mlruns", "--host", "0.0.0.0"]
