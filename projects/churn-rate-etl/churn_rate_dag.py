import csv

from airflow.decorators import dag, task
from airflow.providers.mysql.hooks.mysql import MySqlHook

from datetime import datetime

CSV_DATASET_PATH = '/opt/airflow/dags/df/churn-rate.csv'


@dag(
    dag_id='churn_rate_medallon_graph',
    start_date=datetime(2025, 4, 26),
    schedule='@once',
    catchup=False,
)
def churn_rate_medallon_graph():

    @task(task_id='load_csv_dataset_to_bronze_db', doc_md='Saving CSV raw data to Bronze MySQL DB')
    def load_csv_dataset_to_bronze_db():

        mysql_hook = MySqlHook(mysql_conn_id='mysql_aiven_bronze')
        table = 'churn_rate'

        # Truncate the 'churn_rate' table
        sql = f"TRUNCATE TABLE {table}"
        mysql_hook.run(sql)

        # Load the data from CSV file to 'churn_rate' table
        with open(CSV_DATASET_PATH, 'r', newline='') as f:
            reader = csv.reader(f)

            headers = next(reader)
            rows = [row for row in reader]

            mysql_hook.insert_rows(table=table, rows=rows, target_fields=headers, commit_every=500)

    load_csv_dataset_to_bronze_db()


churn_rate_medallon_graph()
