"""
Here's a simple ELT (Extract, Load, Transform) example using MySQL. In ELT, we:

Extract data from a source (e.g., a CSV or another DB).

Load raw data into a staging table in MySQL.

Transform the data using SQL within MySQL
"""

from datetime import datetime
import faker
import random

from sqlalchemy import (
    create_engine,
    text,
    Table,
    Column,
    Integer,
    String,
    Date,
    DECIMAL,
    MetaData
)
from sqlalchemy.orm import sessionmaker

DB_USER = "avnadmin"
DB_HOST = "mysql-25effa04-oleksandr-45fd.d.aivencloud.com"
DB_PORT = "25202"
DB_NAME = "elt_db"

DB_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)
session = Session()
metadata = MetaData()

# Define tables
staging_sales = Table("staging_sales", metadata,
                      Column("id", Integer, primary_key=True),
                      Column("customer_name", String(100)),
                      Column("amount", DECIMAL(10, 2)),
                      Column("purchase_date", Date)
                      )

sales = Table("sales", metadata,
              Column("sale_id", Integer, primary_key=True),
              Column("customer_name", String(100)),
              Column("amount", DECIMAL(10, 2)),
              Column("purchase_date", Date)
              )

log_table = Table("log_table", metadata,
                  Column("log_id", Integer, primary_key=True, autoincrement=True),
                  Column("message", String(500)),
                  Column("log_time", Date, default=datetime.now())
                  )

# Create tables
# metadata.create_all(engine)

# Generate 10,000 records
faker_gen = faker.Faker()
records = []
for i in range(1, 10001):
    name = faker_gen.name() if random.random() > 0.05 else None  # 5% nulls
    amount = random.uniform(-500, 100000) if random.random() > 0.1 else None  # 10% nulls
    date = faker_gen.date_between(start_date='-2y', end_date='today') if random.random() > 0.1 else None
    records.append((i, name, amount, date))

# Clean previous
with engine.begin() as conn:
    conn.execute(text("DELETE FROM staging_sales"))
    conn.execute(text("DELETE FROM sales"))
    conn.execute(text("DELETE FROM log_table"))

# Insert into staging
with engine.begin() as conn:
    conn.execute(
        text("INSERT INTO staging_sales (id, customer_name, amount, purchase_date) VALUES (:id, :name, :amount, :date)"),
        [{"id": r[0], "name": r[1], "amount": r[2], "date": r[3]} for r in records]
    )

with engine.connect() as conn:
    # Count in staging_sales
    staging_count = conn.execute(text("SELECT COUNT(*) FROM staging_sales")).scalar()

    # Count in sales
    sales_count = conn.execute(text("SELECT COUNT(*) FROM sales")).scalar()

print(f"staging_sales: {staging_count} rows")
print(f"sales: {sales_count} rows")
