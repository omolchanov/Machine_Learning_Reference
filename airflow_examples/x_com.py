# https://khashtamov.com/ru/apache-airflow-xcom/

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator


def print_xcom(**kwargs):
    print('The value is %s' % (kwargs['ti'].xcom_pull(task_ids='bash-echo')))


def push_cmd(**kwargs):
    kwargs['ti'].xcom_push(value='10', key='my_value')


def push_cmd2(**kwargs):
    kwargs['ti'].xcom_push(value='20', key='my_value2')


def pull_cmd(**kwargs):
    print('VALUE: ', kwargs['ti'].xcom_pull(task_ids='push-cmd', key='my_value'))


with DAG(
    dag_id='x_com_example',
    start_date=datetime(2025, 4, 13),
    schedule='@once'
) as dag:

    cmd = BashOperator(
        task_id='bash-echo',
        bash_command='echo "Hello!"',
        # do_xcom_push=False
    )

    printer = PythonOperator(
        task_id='python-printer',
        python_callable=print_xcom
    )

    push_cmd_task = PythonOperator(
        task_id='push-cmd',
        python_callable=push_cmd
    )

    pull_cmd_task = PythonOperator(
        task_id='pull-cmd',
        python_callable=pull_cmd
    )

    push_cmd_task >> pull_cmd_task
