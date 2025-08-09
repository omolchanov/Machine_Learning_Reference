"""
An OLAP cube organizes data along multiple dimensions (like time, geography, product, etc.) and
lets you aggregate (sum, count, avg, etc.) efficiently.
"""
import random
from datetime import date, timedelta

import pandas as pd
from clickhouse_connect import get_client

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)

client = get_client(
    host='localhost',
    port=8123,
    username='testuser',
    password='testuser',
    database='test'
)

if __name__ == '__main__':
    print(client.ping())

    # Create sales table (if not exists)
    client.command('''
    CREATE TABLE IF NOT EXISTS sales (
        date Date,
        region String,
        product String,
        sales UInt32,
        quantity UInt32
    ) ENGINE = MergeTree()
    ORDER BY (date, region, product)
    ''')

    # Clear old data
    client.command('TRUNCATE TABLE sales')

    # Dimension values
    regions = ["North", "South", "East", "West"]
    products = ["ProductA", "ProductB", "ProductC", "ProductD"]

    # Generate data for 60 days
    start_date = date(2025, 6, 1)
    rows = []

    for day_offset in range(60):
        current_date = start_date + timedelta(days=day_offset)
        for region in regions:
            for product in products:
                sales = random.randint(50, 500)  # Random sales amount
                quantity = random.randint(1, 20)  # Random quantity sold
                rows.append((current_date, region, product, sales, quantity))

    client.insert('sales', rows, column_names=['date', 'region', 'product', 'sales', 'quantity'])

    # OLAP Query: Sum of sales and quantity grouped by region and product
    query = '''
    SELECT
        region,
        product,
        sum(sales) AS total_sales,
        sum(quantity) AS total_quantity
    FROM sales
    GROUP BY region, product
    ORDER BY region, product
    '''

    result = client.query(query).result_rows

    print("OLAP Cube Aggregation (sales by region and product):")
    print(client.query(query).column_names)
    for row in result:
        print(row)

    # Advanced OLAP query with CUBE and time breakdown
    query = '''
    SELECT
        toYear(date) AS year,
        toMonth(date) AS month,
        region,
        product,
        SUM(sales) AS total_sales,
        SUM(quantity) AS total_quantity
    FROM sales
    GROUP BY
    GROUPING SETS (
        (year, month, region, product),  -- Detail rows
        (year, month, region),            -- Subtotal by region
        (year, month),                    -- Subtotal by month
        (year)                           -- Subtotal by year
    )
    ORDER BY year, month, region, product
    '''

    result = client.query(query).result_rows
    columns = client.query(query).column_names

    df = pd.DataFrame(result, columns=columns)
    print(df)
