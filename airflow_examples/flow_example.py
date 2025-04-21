# https://stackoverflow.com/questions/43678408/how-to-create-a-conditional-task-in-airflow

from datetime import datetime

from airflow.decorators import dag, task, task_group, branch_task
from airflow.datasets import Dataset

import pandas as pd

csv_dataset = Dataset('/opt/airflow/dags/df/churn-rate.csv')


@dag(
    dag_id='flow_example',
    schedule=[csv_dataset],
    start_date=datetime(2025, 4, 15),
    catchup=False
)
def dataset_flow_example():

    @task_group(group_id='parallel_tasks')
    def run_parallel_tasks():

        @task(task_id='task1')
        def task_1():
            print('I am task 1')
            df = pd.read_csv(csv_dataset.uri)
            print(df.head())

        @task(task_id='task3')
        def task_3():
            print('I am task 3')

        task_1()
        task_3()

    @task_group(group_id='dependent_tasks')
    def run_dependent_tasks():
        @task
        def task_2():
            print('I am task 2')

        @task
        def task_4():
            print('I am task 4')

        task_2() >> task_4()

    run_parallel_tasks() >> run_dependent_tasks()


dataset_flow_example()


@dag(
    dag_id='conditional_flow',
    start_date=datetime(2025, 4, 15),
    schedule='@once',
    catchup=False
)
def conditional_flow_example():

    @task(task_id='task1')
    def task_1(**kwargs):
        print('task1')
        kwargs['ti'].xcom_push(value='15', key='number')

    @branch_task(task_id='branch_task_2')
    def task_2(**kwargs):
        print('task2')

        n = kwargs['ti'].xcom_pull(task_ids='task1', key='number')
        n = int(n)

        print(type(n))

        if n > 10:
            return 'task3'

        elif n < 10:
            return 'task4'

        else:
            return None

    @task(task_id='task3', outlets=[csv_dataset])
    def task_3(**kwargs):
        print('task3')

        n = kwargs['ti'].xcom_pull(task_ids='task1', key='n')
        print(n)

        print(csv_dataset.uri)

    @task(task_id='task4')
    def task_4(**kwargs):
        print('task4')
        n = kwargs['ti'].xcom_pull(task_ids='task1', key='n')
        print(n / 2)

    task_1()
    task_2() >> task_3()
    task_2() >> task_4()


conditional_flow_example()
