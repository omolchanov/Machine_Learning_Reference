import csv
import mysql.connector


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


load_csv_dataset_to_bronze_db()
load_mysql_raw_dataset_to_bronze_db()

