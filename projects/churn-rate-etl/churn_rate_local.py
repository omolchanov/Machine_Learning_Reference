import csv

import mysql.connector
from pymongo import MongoClient

import pandas as pd

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)


# Paths and connections
CSV_DATASET_PATH = '../../assets/churn-rate.csv'

raw_db = mysql.connector.connect(
    host='mysql-25effa04-oleksandr-45fd.d.aivencloud.com',
    user='avnadmin',
    password='',
    port='25202',
    database='churn_rate'
)

bronze_db = mysql.connector.connect(
    host='mysql-25effa04-oleksandr-45fd.d.aivencloud.com',
    user='avnadmin',
    password='',
    port='25202',
    database='churn_rate_bronze'
)

silver_db = mysql.connector.connect(
    host='mysql-25effa04-oleksandr-45fd.d.aivencloud.com',
    user='avnadmin',
    password='',
    port='25202',
    database='churn_rate_silver'
)

gold_db = mysql.connector.connect(
    host='mysql-25effa04-oleksandr-45fd.d.aivencloud.com',
    user='avnadmin',
    password='',
    port='25202',
    database='churn_rate_gold'
)

mongo_db_client = MongoClient(
    'mongodb+srv://oleksandrmolchanov:test123@churn-rate-scam.bnuspoh.mongodb.net/'
    '?retryWrites=true&w=majority&appName=churn-rate-scam')


def load_csv_dataset_to_bronze_db():
    bronze_db_cursor = bronze_db.cursor()

    table = 'Churn_rate'

    # Truncate the 'churn_rate' table
    sql = f"TRUNCATE TABLE {table}"
    bronze_db_cursor.execute(sql)

    # Load the data from CSV file to 'churn_rate' table
    with open(CSV_DATASET_PATH, 'r', newline='') as f:
        reader = csv.reader(f)

        headers = next(reader)
        rows = [row for row in reader]

        sql = f"INSERT INTO {table} ({', '.join(headers)}) VALUES ({', '.join(['%s'] * len(headers))})"
        bronze_db_cursor.executemany(sql, rows)
        bronze_db.commit()

    # bronze_db_cursor.close()
    # bronze_db.close()


def load_mysql_raw_dataset_to_bronze_db():
    raw_db_cursor = raw_db.cursor()
    bronze_db_cursor = bronze_db.cursor()

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
        bronze_db_cursor.execute(sql)

    # Load the data from raw MySql DB to Bronze DB tables
    for i, table in enumerate(tables):
        select_sql = f"SELECT * FROM {table}"
        raw_db_cursor.execute(select_sql)
        rows = raw_db_cursor.fetchall()

        insert_sql = f"INSERT INTO {table} VALUES ({', '.join(['%s'] * len(headers[i]))})"
        bronze_db_cursor.executemany(insert_sql, rows)
        bronze_db.commit()

    # bronze_db_cursor.close()
    # bronze_db.close()

    # raw_db_cursor.close()
    # raw_db.close()


def load_mongodb_anomalies_analysis_to_bronze_db():

    # Get the scam data from MongoDB
    mongo_db = mongo_db_client.churn_rate
    mongo_collection = mongo_db.anomalies_analysis

    rows = []
    mongo_cursor = mongo_collection.find({})
    for d in mongo_cursor:
        d['_id'] = str(d['_id'])
        d.update()

        r = tuple(d.values())
        rows.append(r)

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

    bronze_db_cursor = bronze_db.cursor()

    # Truncate the Bronze DB table
    sql = f"TRUNCATE TABLE {table}"
    bronze_db_cursor.execute(sql)

    insert_sql = f"INSERT INTO {table} ({', '.join(headers)}) VALUES ({', '.join(['%s'] * len(headers))})"
    bronze_db_cursor.executemany(insert_sql, rows)
    bronze_db.commit()

    # bronze_db_cursor.close()
    # bronze_db.close()

    mongo_cursor.close()


def prepare_churn_rate_data_silver_db():
    table = 'Churn_rate'
    select_sql = f"SELECT * FROM {table}"

    df = pd.read_sql(select_sql, bronze_db)

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
    silver_db_cursor = silver_db.cursor()

    sql = f"TRUNCATE TABLE {table}"
    silver_db_cursor.execute(sql)

    rows = list(df.itertuples(index=False, name=None))
    insert_sql = f"INSERT INTO {table} ({', '.join(df.columns)}) VALUES ({', '.join(['%s'] * len(df.columns))})"

    silver_db_cursor.executemany(insert_sql, rows)
    silver_db.commit()

    # silver_db_cursor.close()
    # silver_db.close()


def prepare_customer_data_silver_db():
    silver_db_cursor = silver_db.cursor()
    table = 'Customer'

    sql = f"TRUNCATE TABLE {table}"
    silver_db_cursor.execute(sql)

    # Selecting Customer data from Bronze DB, loading it to Silver DB
    select_sql = (f"SELECT bd_b.* FROM churn_rate_bronze.Customer AS bd_b "
                  f"JOIN churn_rate_silver.Churn_rate AS bd_s "
                  f"ON bd_b.phone_number = bd_s.phone_number")

    silver_db_cursor.execute(select_sql)
    rows = silver_db_cursor.fetchall()

    # Loading Customer table in silver DB with Customer table data from bronze DB
    insert_sql = f"INSERT INTO {table} (id, phone_number, first_name, last_name, churn) VALUES (%s, %s, %s, %s, %s)"
    silver_db_cursor.executemany(insert_sql, rows)
    silver_db.commit()

    # Loading Customer table in silver DB with Customer_Balance table from bronze DB
    select_sql = (f"SELECT bd_b.customer_id, bd_b.balance FROM churn_rate_bronze.Customer_Balance AS bd_b "
                  f"JOIN churn_rate_silver.Customer AS bd_s "
                  f"ON bd_b.customer_id = bd_s.id")

    silver_db_cursor.execute(select_sql)
    rows = silver_db_cursor.fetchall()

    update_sql = f"UPDATE {table} SET balance = %s WHERE id = %s"
    update_data = [(balance, id_) for (id_, balance) in rows]

    silver_db_cursor.executemany(update_sql, update_data)
    silver_db.commit()

    # Loading Customer table in silver DB with Customer_Tenure table data from bronze DB
    select_sql = (f"SELECT bd_b.customer_id, bd_b.join_date, bd_b.churn_date FROM churn_rate_bronze.Customer_Tenure "
                  f"AS bd_b "
                  f"JOIN churn_rate_silver.Customer AS bd_s "
                  f"ON bd_b.customer_id = bd_s.id")

    silver_db_cursor.execute(select_sql)
    rows = silver_db_cursor.fetchall()

    update_sql = f"UPDATE {table} SET join_date = %s, churn_date = %s WHERE id = %s"
    update_data = [(join_date, churn_date, id_) for (id_, join_date, churn_date) in rows]

    silver_db_cursor.executemany(update_sql, update_data)
    silver_db.commit()

    # silver_db_cursor.close()
    # silver_db.close()


def prepare_churn_rate_report_data_golden_db():
    silver_db_cursor = silver_db.cursor()
    gold_db_cursor = gold_db.cursor()

    table = 'Churn_rate_Report'

    sql = f"TRUNCATE TABLE {table}"
    gold_db_cursor.execute(sql)

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
    silver_db_cursor.execute(select_sql)
    rows = silver_db_cursor.fetchall()

    headers_sql = f"SHOW COLUMNS FROM {table}"
    gold_db_cursor.execute(headers_sql)
    headers_res = gold_db_cursor.fetchall()
    headers = [h[0] for h in headers_res][1:]

    insert_sql = f"INSERT INTO {table} ({', '.join(headers)}) VALUES ({', '.join(['%s'] * len(headers))})"
    gold_db_cursor.executemany(insert_sql, rows)
    gold_db.commit()

    silver_db_cursor.close()
    silver_db.close()

    gold_db_cursor.close()
    gold_db.close()


load_csv_dataset_to_bronze_db()
load_mysql_raw_dataset_to_bronze_db()
load_mongodb_anomalies_analysis_to_bronze_db()
prepare_churn_rate_data_silver_db()
prepare_customer_data_silver_db()
prepare_churn_rate_report_data_golden_db()
