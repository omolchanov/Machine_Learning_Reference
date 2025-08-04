from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule

from datetime import timedelta, datetime
import random
import logging
import pickle
import os

# Functions
from funcs import *

default_args = {
    'owner': 'mlops_user',
    'retry_delay': timedelta(minutes=5),
}


@dag(
    dag_id='sales_mlops_pipeline',
    default_args=default_args,
    description='Sales MLOps DAG',
    schedule_interval='@once',
    start_date=days_ago(1),
    catchup=False,
)
def sales_mlops_pipeline():
    @task
    def extract_data():
        data = func_extract_data()
        logging.info(f"The data was extracted")

        return data

    @task
    def validate_data(data):
        min_shape = 5000  # Minimum shape of the dataset

        if len(data) < min_shape:
            raise ValueError(f"Data validation failed: insufficient data.Size of the dataset is {data.shape}. "
                             f"Minimum size is {min_shape}")

        logging.info(f"Data was validated. Size of the dataset is {data.shape}")
        return True

    @task()
    def preprocess_data(data):
        return func_preprocess_data(data)

    @task()
    def train_model(processed_data):
        func_train_model(processed_data)

    @task()
    def track_experiment(data):
        func_conduct_experiment(data)

    @task(
        trigger_rule=TriggerRule.ALL_SUCCESS,
        doc_md="Goal: Ensure model meets quality standards."
               "Activities: Cross-validation, hold-out testing."
               "Performance vs baseline, fairness testing."
    )
    def evaluate_model(model):
        print("Evaluating model...")
        accuracy = model["coef"]
        print(f"Model accuracy: {accuracy:.2f}")
        if accuracy < 0.5:
            raise ValueError("Model accuracy below threshold!")
        return accuracy

    @task(
        trigger_rule=TriggerRule.ALL_SUCCESS,
        doc_md="Goal: Prepare model for deployment in a reproducible way. "
               "Activities: Serialize to .pkl, .onnx, .joblib, etc. "
               "Create Python/REST/GRPC interface for use."
    )
    def package_model(model):
        print("Packaging model...")

        # Simulate saving a model file
        with open("/tmp/model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("Model saved to model.pkl")

        return "model.pkl"

    @task(
        trigger_rule=TriggerRule.ALL_SUCCESS,
        doc_md=""
               "Goal: Serve model in production for inference. "
               "Activities: Deploy to REST API, batch jobs, or streaming. "
               "Monitor performance and resource use."
    )
    def deploy_model(model):
        print(f"Deploying model with coef = {model['coef']:.2f}...")

    @task(
        trigger_rule=TriggerRule.ALL_SUCCESS,
        doc_md="Goal: Detect drift, degradation, or failures in production. "
               "Activities: Track inputs, outputs, latency, errors. "
               "Detect data drift or model decay."
    )
    def monitor_and_log(model):
        print("Monitoring model performance in production...")
        # Simulated monitoring logs
        print("No drift detected. Latency normal. Error rate < 1%.")

    # DAG execution flow
    data = extract_data()
    validate_data(data)
    processed_data = preprocess_data(data)
    train_model(processed_data)
    track_experiment(processed_data)
    # evaluate_model(model)
    # package_model(model)
    # deploy_model(model)
    # monitor_and_log(model)


sales_mlops_pipeline()
