from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
from datetime import datetime, timedelta
import logging

from funcs import *


@dag(
    dag_id='sales_mlops_monitoring',
    start_date=datetime(2025, 8, 1),
    schedule_interval=timedelta(minutes=1),
    catchup=False,
)
def sales_mlops_monitoring():
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

    @task
    def evaluate_model(data, model):
        logging.info(f"Loaded model {model}")

        scores = func_evaluate_model(data, model)
        mean_score, roc_auc_score = scores

        mean_score_thr = 0.9
        roc_auc_score_thr = 0.9

        if mean_score < mean_score_thr or roc_auc_score < roc_auc_score_thr:
            raise AirflowSkipException(f""
                                       f"Model evalution failed and it can not be used in PROD "
                                       f"with acc_score {mean_score:.3f} and roc_auc {roc_auc_score:.3f}")

        logging.info(f"The model was evaluated successfuly and can be used in PROD "
                     f"with acc_score {mean_score:.3f} and roc_auc {roc_auc_score:.3f}")

    data = extract_data()
    validate_data(data)

    data = preprocess_data(data)
    model = load_prod_model()

    evaluate_model(data, model)


sales_mlops_monitoring()
