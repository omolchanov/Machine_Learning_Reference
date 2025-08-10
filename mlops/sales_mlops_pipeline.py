from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule

from datetime import timedelta, datetime
import random
import logging
import pickle
import os
import joblib
import base64
import io

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
        model = func_train_model(processed_data)

        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    @task()
    def conduct_experiment(data, model):
        model_bytes = base64.b64decode(model)
        buffer = io.BytesIO(model_bytes)
        model = joblib.load(buffer)

        func_conduct_experiment(data, model)

    @task(trigger_rule=TriggerRule.ALL_SUCCESS,)
    def evaluate_model(data, model):
        model_bytes = base64.b64decode(model)
        buffer = io.BytesIO(model_bytes)
        model = joblib.load(buffer)

        scores = func_evaluate_model(data, model)
        mean_score, roc_auc_score = scores

        mean_score_thr = 0.5
        roc_auc_score_thr = 0.2

        if mean_score < mean_score_thr or roc_auc_score < roc_auc_score_thr:
            raise ValueError(f""
                             f"Model evalution failed and it can not be used in PROD "
                             f"with acc_score {mean_score:.3f} and roc_auc {roc_auc_score:.3f}")

        logging.info(f"The model was evaluated successfuly and can be used in PROD "
                     f"with acc_score {mean_score:.3f} and roc_auc {roc_auc_score:.3f}")

    @task(trigger_rule=TriggerRule.ALL_SUCCESS)
    def package_model(model):
        model_bytes = base64.b64decode(model)
        buffer = io.BytesIO(model_bytes)
        model = joblib.load(buffer)

        func_package_model(model)

    @task(trigger_rule=TriggerRule.ALL_SUCCESS)
    def deploy_model():
        func_deploy_model()

    # DAG execution flow
    data = extract_data()
    validate_data(data)

    data = preprocess_data(data)
    model = train_model(data)

    (conduct_experiment(data, model)
     >> evaluate_model(data, model)
     >> package_model(model)
     >> deploy_model())


sales_mlops_pipeline()
