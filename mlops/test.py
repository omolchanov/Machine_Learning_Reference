"""
MLOps Lifecycle: Stages Overview

🧪 1. Data Ingestion & Validation
Goal: Collect raw data and check quality before use.
Activities: Extract data from APIs, DBs, files, etc. Validate schemas, check nulls, detect drift/anomalies.
Tools: Airflow, Prefect, Great Expectations, Pandera

🧹 2. Data Preprocessing & Feature Engineering
Goal: Clean, transform, and enrich data for modeling.
Activities: Handle missing values, outliers. Encode categories, normalize, scale features.
Tools: pandas, scikit-learn, Spark, dbt (for SQL-based)

🧠 3. Model Training
Goal: Train one or more ML models using prepared data.
Activities: Experiment with different algorithms. Log hyperparameters, results.
Tools: scikit-learn, XGBoost, TensorFlow, PyTorch

🧾 4. Experiment Tracking
Goal: Record and compare model runs and metrics.
Activities: Store training metadata (params, metrics, version, etc.)
Tools: MLflow, Weights & Biases, Neptune.ai

🧪 5. Model Validation & Evaluation
Goal: Ensure model meets quality standards.
Activities Cross-validation, hold-out testing.
Performance vs baseline, fairness testing.
Tools: sklearn.metrics, Evidently AI

📦 6. Model Packaging
Goal: Prepare model for deployment in a reproducible way.
Activities: Serialize to .pkl, .onnx, .joblib, etc.
Create Python/REST/GRPC interface for use.
Tools: Docker, Conda, BentoML, FastAPI

🚀 7. Model Deployment
Goal: Serve model in production for inference.
Activities: Deploy to REST API, batch jobs, or streaming. Monitor performance and resource use.
Tools: FastAPI, Flask, Kubernetes, Seldon, SageMaker

📊 8. Monitoring & Logging
Goal: Detect drift, degradation, or failures in production.
Activities: Track inputs, outputs, latency, errors. Detect data drift or model decay.
Tools: Prometheus + Grafana, MLflow, Evidently, Sentry

🔁 9. Continuous Training (CT) / Retraining
Goal: Re-train model periodically or on trigger.
Activities: Detect drift or schedule retraining. Automatically re-train + re-deploy if approved.
Tools: Airflow, Prefect, Kubeflow Pipelines

🧪 10. CI/CD for ML
Goal: Automate the full ML pipeline and validation.
Activities: Unit tests, data tests, model tests. Auto-trigger pipeline on Git commit.
Tools: GitHub Actions, GitLab CI, Jenkins, DVC, Terraform

🧱 Bonus: Optional but useful layers
Feature Store: Centralized management of features (e.g., Feast)
Model Registry: Versioning and lifecycle tracking (e.g., MLflow, SageMaker Model Registry)
Metadata Store: Store lineage, artifacts, and experiment data
"""

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule

from datetime import timedelta
import random
import logging
import pickle
import os

default_args = {
    'owner': 'mlops_user',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


@dag(
    dag_id='simple_mlops_pipeline',
    default_args=default_args,
    description='A simple MLOps DAG using Airflow 2.x TaskFlow API',
    schedule_interval='@once',
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'example'],
)
def mlops_pipeline():
    @task(
        doc_md="Goal: Collect raw data and check quality before use. "
               "Activities: Extract data from APIs, DBs, files, etc. "
               "Validate schemas, check nulls, detect drift/anomalies.")
    def extract_data():
        print("Extracting data...")
        return [random.randint(0, 100) for _ in range(100)]

    @task
    def validate_data(data):
        print("Validating data...")
        if len(data) < 50:
            raise ValueError("Data validation failed: insufficient data.")
        return True

    @task(
        doc_md="Goal: Clean, transform, and enrich data for modeling. "
               "Activities: Handle missing values, outliers."
               "Encode categories, normalize, scale features."
    )
    def preprocess_data(data):
        print("Preprocessing data...")
        return [x / 100 for x in data]

    @task(
        doc_md="Goal: Train one or more ML models using prepared data."
               "Activities: Experiment with different algorithms. "
               "Log hyperparameters, results."
    )
    def train_model(processed_data):
        print("Training model...")
        coef = sum(processed_data) / len(processed_data)
        params = {"learning_rate": 0.1}  # Example hyperparameter
        score = coef  # Dummy metric
        return {"coef": coef, "params": params, "score": score}

    @task(
        doc_md="Goal: Track model experiments and artifacts. "
               "Activities: Log model parameters, metrics, artifacts using tools like MLflow or Weights & Biases."
    )
    def track_experiment(model):
        print("Tracking experiment...")
        logging.info(
            f"Logging experiment: coef={model['coef']:.2f}, params={model['params']}, score={model['score']:.2f}")

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
    processed = preprocess_data(data)
    model = train_model(processed)
    track_experiment(model)
    evaluate_model(model)
    package_model(model)
    deploy_model(model)
    monitor_and_log(model)


mlops_pipeline()
