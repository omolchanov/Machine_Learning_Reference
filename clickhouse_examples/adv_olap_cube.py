from datetime import datetime, timedelta
import random

from clickhouse_connect import get_client
import pandas as pd

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
    CREATE TABLE IF NOT EXISTS purchase (
        purchase_date Date,
        g_year UInt16,
        g_month UInt8,
        g_day UInt8,
        region String,
        product String,
        sales UInt32,
        quantity UInt32
    ) ENGINE = MergeTree()
    ORDER BY (purchase_date, region, product)
    ''')

    # Generate sample data for 60 days
    regions = ['North', 'South', 'East', 'West']
    products = ['Laptop', 'Phone', 'Tablet', 'Monitor']

    start_date = datetime(2024, 1, 1)
    data = []
    for day_offset in range(60):
        current_date = start_date + timedelta(days=day_offset)
        for _ in range(random.randint(5, 10)):
            data.append((
                current_date.date(),
                current_date.year,
                current_date.month,
                current_date.day,
                random.choice(regions),
                random.choice(products),
                random.randint(100, 1000),  # sales
                random.randint(1, 20)  # quantity
            ))

    client.insert(
        'purchase',
        data,
        column_names=['purchase_date', 'g_year', 'g_month', 'g_day', 'region', 'product', 'sales', 'quantity']
    )

    query = '''
    SELECT
        if(grouping(g_year) = 1, 'ALL', toString(g_year)) AS year,
        if(grouping(g_month) = 1, 'ALL', toString(g_month)) AS month,
        if(grouping(g_day) = 1, 'ALL', toString(g_day)) AS day,
        if(grouping(region) = 1, 'ALL', region) AS g_region,
        if(grouping(product) = 1, 'ALL', product) AS g_product,
        sum(sales) AS total_sales,
        sum(quantity) AS total_quantity,
        grouping(g_year) + grouping(g_month) + grouping(g_day) + grouping(region) + grouping(product) AS subtotal_level
    FROM purchase
    GROUP BY CUBE(
        toYear(purchase_date) AS g_year,
        toMonth(purchase_date) AS g_month,
        toDayOfMonth(purchase_date) AS g_day,
        region AS region,
        product AS product
    )
    ORDER BY year, month, day, region, product
    '''

    rows = client.query(query).result_rows
    columns = client.query(query).column_names

    df = pd.DataFrame(rows, columns=columns)
    print(df.head(50))

    # Top popular
    query = '''
    SELECT *
    FROM (
        SELECT
            toYear(purchase_date) AS g_year,
            region,
            product,
            SUM(sales) AS total_sales,
            RANK() OVER (PARTITION BY toYear(purchase_date), region ORDER BY SUM(sales) DESC) AS sales_rank
        FROM purchase
        GROUP BY g_year, region, product
    ) AS ranked
    WHERE sales_rank <= 3
    ORDER BY g_year, region, sales_rank
    '''

    rows = client.query(query).result_rows
    columns = client.query(query).column_names

    df = pd.DataFrame(rows, columns=columns)
    print(df)

    """
    We group by year, month, region, product to analyze monthly product sales by region.
    moving_avg_7days: Smooth sales trends over the last 7 months (can adjust window frame).
    sales_rank: Rank products within region-month to find top 5.
    pct_sales_region: Product sales as % of region sales for month.
    sales_gt_500: Conditional sum counting only sales > 500.

    HAVING sales_rank <= 5 filters only top 5 products per region-month.
    """

    query = '''
    SELECT
        g_year,
        g_month,
        region,
        product,
        total_sales,
        total_quantity,

    -- Moving average of total_sales over last 7 months for each product-region
    avg(total_sales) OVER (
        PARTITION BY region, product
        ORDER BY g_year, g_month
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7months,

    -- Ranking products by total_sales per year-month-region
    sales_rank,

    -- Percentage sales contribution per region-month
    total_sales * 100.0 / sum(total_sales) OVER (
        PARTITION BY g_year, g_month, region
    ) AS pct_sales_region,

    -- Sum of sales > 500 conditionally (from total_sales here)
    CASE WHEN total_sales > 500 THEN total_sales ELSE 0 END AS sales_gt_500

    FROM (
        SELECT
            toYear(purchase_date) AS g_year,
            toMonth(purchase_date) AS g_month,
            region,
            product,
            sum(sales) AS total_sales,
            sum(quantity) AS total_quantity,
    
            -- Rank after aggregation
            RANK() OVER (
                PARTITION BY toYear(purchase_date), toMonth(purchase_date), region
                ORDER BY sum(sales) DESC
            ) AS sales_rank
    
        FROM purchase
        GROUP BY
            g_year,
            g_month,
            region,
            product
    ) AS agg
    WHERE sales_rank <= 5
    ORDER BY
        g_year,
        g_month,
        region,
        sales_rank

    '''

    rows = client.query(query).result_rows
    columns = client.query(query).column_names

    df = pd.DataFrame(rows, columns=columns)
    print(df)
