import csv
import pprint

from airflow.decorators import dag, task
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.providers.mongo.hooks.mongo import MongoHook
from airflow.datasets import Dataset

from datetime import datetime

import pandas as pd
import numpy as np

# Configuration
np.set_printoptions(threshold=np.inf, suppress=True)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)

# Paths and connections
CSV_DATASET_PATH = '/opt/airflow/dags/df/churn-rate.csv'

raw_mysql_db = MySqlHook(mysql_conn_id='mysql_aiven_raw')
bronze_db = MySqlHook(mysql_conn_id='mysql_aiven_bronze')
silver_db = MySqlHook(mysql_conn_id='mysql_aiven_silver')
mongo_db = MongoHook(mongo_conn_id='churn_rate_mongo')


# Datasets
csv_dataset = Dataset(CSV_DATASET_PATH)
raw_mysql_db_dataset = Dataset('Raw_MySql_DB')
mongo_db_dataset = Dataset('MongoDB_Scam')

bronze_db_dataset = Dataset('Churn_rate_Bronze_DB')
silver_db_dataset = Dataset('Churn_rate_Silver_DB')


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

    @task(
        task_id='load_mongodb_anomalies_analysis_to_bronze_db',
        doc_md='Saving MongoDB anomalies detection data to Bronze MySQL DB',
        outlets=[mongo_db_dataset]
    )
    def load_mongodb_anomalies_analysis_to_bronze_db():
        client = mongo_db.get_conn()
        db = client.churn_rate
        anomalies_analysis_collection = db.anomalies_analysis

        print(anomalies_analysis_collection)

        rows = []
        mongo_cursor = anomalies_analysis_collection.find({})
        for d in mongo_cursor:
            d['_id'] = str(d['_id'])
            d.update()

            r = tuple(d.values())
            rows.append(r)

        # Load data to the bronze DB
        table = 'Anomalies_Analysis'

        headers = [
            'mongo_id',
            'customer_id',
            'phone_number',
            'outgoing_phone_number',
            'timestamp',
            'fraud_scoring',
            'is_fraud'
        ]

        # Truncate the Bronze DB table
        sql = f"TRUNCATE TABLE {table}"
        bronze_db.run(sql)

        bronze_db.insert_rows(table=table, rows=rows, target_fields=headers, commit_every=500)

    @task(
        task_id='prepare_churn_rate_data_silver_db',
        doc_md='Removing duplicates and empty rows from Churn_rate table and saving to Silver DB',
        outlets=[bronze_db_dataset, silver_db_dataset]
    )
    def prepare_churn_rate_data_silver_db():
        table = 'Churn_rate'
        select_sql = f"SELECT * FROM {table}"

        df:pd.DataFrame = bronze_db.get_pandas_df(sql=select_sql)

        # Remove unused columns
        df = df.drop(['area_code'], axis=1)

        # Remove duplicates
        print('===REMOVING DUPLICATES===')
        df.drop_duplicates(inplace=True)

        # Remove empty values
        print('===REMOVING EMPTY VALUES===')
        df = df.fillna(0)

        # Saving the pre-processed data to silver DB
        # Truncating Churn_rate table

        sql = f"TRUNCATE TABLE {table}"
        silver_db.run(sql)

        rows = df.to_records(index=False).tolist()
        silver_db.insert_rows(table=table, rows=rows, target_fields=df.columns.tolist(), commit_every=500)

    [
        load_csv_dataset_to_bronze_db(),
        load_mysql_raw_dataset_to_bronze_db(),
        load_mongodb_anomalies_analysis_to_bronze_db()
    ] >> prepare_churn_rate_data_silver_db()


churn_rate_medallon_graph()
