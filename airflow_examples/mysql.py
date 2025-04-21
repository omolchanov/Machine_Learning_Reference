# https://www.restack.io/docs/airflow-knowledge-apache-mysql-providers-connection-hook-pypi-pip
# https://www.astronomer.io/docs/learn/airflow-sql/

import pprint
from datetime import datetime

from airflow.decorators import dag, task
from airflow.providers.mysql.hooks.mysql import MySqlHook


@dag(
    dag_id='mysql_example',
    start_date=datetime(2025, 4, 13),
    schedule='@once',
    catchup=False,
)
def mysql_example_flow():

    @task(task_id='select_raw_data')
    def select_data(**kwargs):
        mysql_hook = MySqlHook(mysql_conn_id='mysql_aiven_raw')

        sql = """SELECT state FROM churn_rate LIMIT 10"""
        res = mysql_hook.get_records(sql)

        pprint.pp(res)
        kwargs['ti'].xcom_push(value=res, key='raw_states')

    @task(task_id='save_raw_data_to_bronze')
    def save_data_to_bronze(**kwargs):
        raw_values = kwargs['ti'].xcom_pull(task_ids='select_raw_data', key='raw_states')
        values = ", ".join(["('" + v[0] + "')" for v in raw_values])

        mysql_hook = MySqlHook(mysql_conn_id='mysql_aiven_bronze')
        sql = """INSERT INTO churn_rate(state) VALUES """ + str(values) + """"""

        print(sql)
        mysql_hook.run(sql)

    select_data() >> save_data_to_bronze()


mysql_example_flow()
