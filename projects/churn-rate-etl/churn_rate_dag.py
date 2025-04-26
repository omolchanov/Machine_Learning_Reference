import csv
import pprint

from airflow.decorators import dag, task
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.datasets import Dataset

from datetime import datetime

# Paths and connections
CSV_DATASET_PATH = '/opt/airflow/dags/df/churn-rate.csv'

raw_mysql_db = MySqlHook(mysql_conn_id='mysql_aiven_raw')
bronze_db = MySqlHook(mysql_conn_id='mysql_aiven_bronze')

# Datasets
csv_dataset = Dataset(CSV_DATASET_PATH)
raw_mysql_db_dataset = Dataset('Raw_MySql_DB')

bronze_db_dataset = Dataset('Churn_rate_Bronze_DB')


@dag(
    dag_id='churn_rate_medallon_graph',
    start_date=datetime(2025, 4, 26),
    schedule='@once',
    catchup=False,
)
def churn_rate_medallon_graph():

    @task(
        task_id='load_csv_dataset_to_bronze_db',
        doc_md='Saving CSV raw data to Bronze MySQL DB',
        outlets=[csv_dataset, bronze_db_dataset]
    )
    def load_csv_dataset_to_bronze_db():
        table = 'Churn_rate'

        # Truncate the 'churn_rate' table
        sql = f"TRUNCATE TABLE {table}"
        bronze_db.run(sql)

        # Load the data from CSV file to 'churn_rate' table
        with open(CSV_DATASET_PATH, 'r', newline='') as f:
            reader = csv.reader(f)

            headers = next(reader)
            rows = [row for row in reader]

            bronze_db.insert_rows(table=table, rows=rows, target_fields=headers, commit_every=500)

    @task(
        task_id='load_mysql_raw_dataset_to_bronze_db',
        doc_md='Saving MySQL raw Customer data to Bronze MySQL DB',
        outlets=[raw_mysql_db_dataset, bronze_db_dataset]
    )
    def load_mysql_raw_dataset_to_bronze_db():
        tables = [
            'Customer',
            'Customer_Balance',
            'Customer_Tenure'
        ]

        headers = [
            ['id', 'phone_number', 'first_name', 'last_name', 'churn'],
            ['id', 'customer_id', 'balance'],
            ['id', 'customer_id', 'join_date', 'churn_date']
        ]

        # Truncate the Bronze DB tables
        for _,table in enumerate(tables):
            sql = f"TRUNCATE TABLE {table}"
            bronze_db.run(sql)

        # Load the data from raw MySql DB to Bronze DB tables
        for i, table in enumerate(tables):

            select_sql = f"SELECT * FROM {table}"
            rows = raw_mysql_db.get_records(select_sql)

            bronze_db.insert_rows(table=table, rows=rows, target_fields=headers[i], commit_every=500)

    load_csv_dataset_to_bronze_db()
    load_mysql_raw_dataset_to_bronze_db()


churn_rate_medallon_graph()
