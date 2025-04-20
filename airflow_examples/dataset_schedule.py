# https://www.astronomer.io/docs/learn/airflow-datasets/?tab=taskflow#dataset-concepts

from datetime import datetime
import csv

from airflow.decorators import dag, task
from airflow.datasets import Dataset

import pandas as pd

CSV_DATASET_URI = '/opt/airflow/dags/df/churn-rate.csv'


@dag(
    dag_id='dataset_scheduler',
    start_date=datetime(2025, 4, 15),
    schedule=[Dataset(CSV_DATASET_URI)],
    catchup=False
)
def dataset_schedule_flow():

    @task
    def notify_on_update():
        df = pd.read_csv(CSV_DATASET_URI)
        print(df.shape)
        print('The dataset was updated')

    notify_on_update()


dataset_schedule_flow()


@dag(
    dag_id='dataset_updater',
    start_date=datetime(2025, 4, 15),
    schedule='@once',
    fail_stop=True
)
def dataset_update_flow():

    @task(
        outlets=[Dataset(CSV_DATASET_URI)],
        show_return_value_in_logs=False
    )
    def update_dataset():
        print('Updating the dataset')

        df = pd.read_csv(CSV_DATASET_URI)
        print(df.shape)

        with open(CSV_DATASET_URI, 'a') as f:
            row = [
                'VT, 100,408,340-9449,yes,no,0,219.4,112,37.3,225.7,102,19.18,255.3,95,11.49,12.0,4,3.24,4,False'
            ]

            writer = csv.writer(f)
            writer.writerow(row)

            f.close()

        df = pd.read_csv(CSV_DATASET_URI)
        print(df.shape)

    update_dataset()


dataset_update_flow()
