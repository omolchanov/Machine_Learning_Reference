# https://khashtamov.com/ru/apache-airflow-xcom/

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def calc_fn(**kwargs):
    tasks = [f'push_{i}' for i in range(1, 5)]
    values = kwargs['ti'].xcom_pull(task_ids=tasks)
    return sum(values)


with DAG(
    dag_id='multiple_x_com',
    start_date=datetime(2025, 4, 13),
    schedule='@once'
) as dag:

    tasks = []

    for i in range(1, 5):
        task = PythonOperator(
            task_id='push_'+str(i),
            python_callable=lambda i=i: i
        )

        tasks.append(task)

    calculator = PythonOperator(
        task_id='calculator',
        python_callable=calc_fn
    )

    calculator.set_upstream(tasks)
