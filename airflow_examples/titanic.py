import warnings
warnings.filterwarnings('ignore')

import requests
from datetime import datetime, timedelta
from pathlib import Path

from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.decorators import task

import pandas as pd


def download_data():
    url = 'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv'
    Path('titanic.csv').write_text(requests.get(url).content.decode())


def pivot_data():
    df = pd.read_csv('titanic.csv')
    df = df.pivot_table(index='Sex', columns=['Survived', 'Pclass'], values='Name', aggfunc='count')

    print(df)


with (DAG(
        dag_id='titanic_dag_1',
        schedule='@once',
        start_date=datetime(2025, 4, 12),
        description='Simple pipeline with Titanic dataframe'
) as dag):

    download_data_task = PythonOperator(
       task_id='download_data',
       python_callable=download_data,
       dag=dag
   )

    pivot_data_task = PythonOperator(
        task_id='pivot_data',
        python_callable=pivot_data,
        dag=dag
    )

download_data_task.doc_md = \
"""
Here is some documentation for Task
"""

dag.doc_md = \
"""
Here is some documentation for DAG
"""

download_data_task >> pivot_data_task
