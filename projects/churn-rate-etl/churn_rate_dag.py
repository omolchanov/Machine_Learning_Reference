import csv

from airflow.decorators import dag, task
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.providers.mongo.hooks.mongo import MongoHook
from airflow.datasets import Dataset

from datetime import datetime

import pandas as pd

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)

# Paths and connections
CSV_DATASET_PATH = '/opt/airflow/dags/df/churn-rate.csv'

raw_mysql_db = MySqlHook(mysql_conn_id='mysql_aiven_raw')
bronze_db = MySqlHook(mysql_conn_id='mysql_aiven_bronze')
silver_db = MySqlHook(mysql_conn_id='mysql_aiven_silver')
gold_db = MySqlHook(mysql_conn_id='mysql_aiven_gold')
mongo_db = MongoHook(mongo_conn_id='churn_rate_mongo')


# Datasets
csv_dataset = Dataset(CSV_DATASET_PATH)
raw_mysql_db_dataset = Dataset('Raw_MySql_DB')
mongo_db_dataset = Dataset('MongoDB_Scam')

bronze_db_dataset = Dataset('Churn_rate_Bronze_DB')
silver_db_dataset = Dataset('Churn_rate_Silver_DB')
gold_db_dataset = Dataset('Churn_rate_Golden_DB')


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

    @task(
        task_id='prepare_customer_data_silver_db',
        doc_md='Aggregates all Customer data from Bronze DB to a single table in Silver DB',
        outlets=[bronze_db_dataset, silver_db_dataset]
    )
    def prepare_customer_data_silver_db():

        # Truncate the Silver DB table
        table = 'Customer'

        sql = f"TRUNCATE TABLE {table}"
        silver_db.run(sql)

        # Selecting Customer data from Bronze DB, loading it to Silver DB
        select_sql = (f"SELECT bd_b.* FROM churn_rate_bronze.Customer AS bd_b "
                      f"JOIN churn_rate_silver.Churn_rate AS bd_s "
                      f"ON bd_b.phone_number = bd_s.phone_number")
        rows = silver_db.get_records(select_sql)

        headers = ['id', 'phone_number', 'first_name', 'last_name', 'churn']
        silver_db.insert_rows(table=table, rows=rows, target_fields=headers, commit_every=500)

        # Loading Customer table in silver DB with Customer_Balance table from bronze DB
        select_sql = (f"SELECT bd_b.customer_id, bd_b.balance FROM churn_rate_bronze.Customer_Balance AS bd_b "
                      f"JOIN churn_rate_silver.Customer AS bd_s "
                      f"ON bd_b.customer_id = bd_s.id")

        rows = silver_db.get_records(select_sql)

        update_sql = f"UPDATE {table} SET balance = %s WHERE id = %s"
        update_data = [(balance, id_) for (id_, balance) in rows]

        for r in update_data:
            silver_db.run(update_sql, parameters=r)

        # Loading Customer table in silver DB with Customer_Tenure table data from bronze DB
        select_sql = (f"SELECT bd_b.customer_id, bd_b.join_date, bd_b.churn_date "
                      f"FROM churn_rate_bronze.Customer_Tenure AS bd_b "
                      f"JOIN churn_rate_silver.Customer AS bd_s "
                      f"ON bd_b.customer_id = bd_s.id")

        rows = silver_db.get_records(select_sql)

        update_sql = f"UPDATE {table} SET join_date = %s, churn_date = %s WHERE id = %s"
        update_data = [(join_date, churn_date, id_) for (id_, join_date, churn_date) in rows]

        for r in update_data:
            silver_db.run(update_sql, parameters=r)

    @task(
        task_id='prepare_churn_rate_report_data_golden_db',
        doc_md='Loading Churn_rate_Report in golden DB from silver DB database',
        outlets=[silver_db_dataset, gold_db_dataset]
    )
    def prepare_churn_rate_report_data_golden_db():

        table = 'Churn_rate_Report'

        sql = f"TRUNCATE TABLE {table}"
        gold_db.run(sql)

        # Loading Churn_rate_Report in golden DB from silver DB database
        select_sql = (f"SELECT tb_chr.*, "
                      f"tb_cust.first_name, "
                      f"tb_cust.last_name, "
                      f"tb_cust.balance, "
                      f"tb_cust.join_date, "
                      f"tb_cust.churn_date FROM Churn_rate AS tb_chr "
                      f"JOIN Customer AS tb_cust "
                      f"ON tb_chr.phone_number = tb_cust.phone_number "
                      f"WHERE tb_cust.join_date IS NOT NULL OR tb_cust.balance IS NOT NULL")

        rows = silver_db.get_records(select_sql)

        headers_sql = f"SHOW COLUMNS FROM {table}"
        headers_res = gold_db.get_records(headers_sql)
        headers = [h[0] for h in headers_res][1:]

        gold_db.insert_rows(table=table, rows=rows, target_fields=headers, commit_every=500)

    @task(
        task_id='prepare_anomalies_analysis_silver_db',
        doc_md='Loads Anomalies analysis data from Bronze DB to Silver DB',
        outlets=[bronze_db_dataset, silver_db_dataset]
    )
    def prepare_anomalies_analysis_silver_db():

        table = 'Anomalies_Analysis'
        headers = ['customer_id', 'phone_number', 'outgoing_phone_number', 'timestamp', 'fraud_scoring', 'is_fraud']

        # Truncate table
        sql = f"TRUNCATE TABLE {table}"
        silver_db.run(sql)

        # Load Anomalies analysis data from Bronze DB to Silver DB
        select_sql = f"SELECT {', '.join(headers)} FROM Anomalies_Analysis"
        rows = bronze_db.get_records(select_sql)

        silver_db.insert_rows(table=table, rows=rows, target_fields=headers, commit_every=500)

    @task(
        task_id='prepare_anomalies_analysis_golden_db',
        doc_md='Loads Anomalies Analysis Data from silver DB to golden DB',
        outlets=[silver_db_dataset, gold_db_dataset]
    )
    def prepare_anomalies_analysis_golden_db():

        table = 'Anomalies_Analysis_Report'
        headers = ['customer_id', 'phone_number', 'outgoing_phone_number', 'timestamp', 'fraud_scoring', 'is_fraud']

        # Truncate table
        sql = f"TRUNCATE TABLE {table}"
        gold_db.run(sql)

        # Loads Anomalies Analysis Data from silver DB to golden DB
        select_sql = (f"SELECT {', '.join(headers)} FROM churn_rate_silver.Anomalies_Analysis "
                      f"WHERE customer_id "
                      f"IN (SELECT id FROM churn_rate_gold.Churn_rate_Report)")
        rows = silver_db.get_records(select_sql)

        gold_db.insert_rows(table=table, rows=rows, target_fields=headers, commit_every=500)

    ([
        load_csv_dataset_to_bronze_db(),
        load_mysql_raw_dataset_to_bronze_db(),
        load_mongodb_anomalies_analysis_to_bronze_db()
    ] >>

     prepare_churn_rate_data_silver_db() >>
     prepare_customer_data_silver_db() >>
     prepare_anomalies_analysis_silver_db() >>

     prepare_churn_rate_report_data_golden_db() >>
     prepare_anomalies_analysis_golden_db())


churn_rate_medallon_graph()
