"""
Sales Trend Analysis

Dataset: Sales records with timestamps
Goal: Analyze monthly/weekly trends, seasonal patterns
Tools: SQL GROUP BY, time-based windowing, line plots
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)


if __name__ == '__main__':

    # Sample sales data
    data = {
        "sale_id": range(1, 101),
        "sale_date": pd.date_range(start="2024-01-01", periods=100, freq='D'),
        "amount": (100 + (np.random.rand(100) * 900)).round(2)
    }
    sales_df = pd.DataFrame(data)

    # Connect to duckdb
    con = duckdb.connect()
    con.register('sales', sales_df)

    # SQL: Monthly sales trend (sum amount grouped by year-month)
    monthly_sales = con.execute("""
        SELECT 
            STRFTIME(sale_date, '%Y-%m') AS year_month,
            SUM(amount) AS total_sales
        FROM sales
        GROUP BY year_month
        ORDER BY year_month
    """).fetch_df()
    print(monthly_sales)

    # SQL: Weekly sales trend (sum amount grouped by year-week)
    weekly_sales = con.execute("""
        SELECT
            STRFTIME(sale_date, '%Y-%W') AS year_week,
            SUM(amount) AS total_sales
        FROM sales
        GROUP BY year_week
        ORDER BY year_week
    """).fetch_df()
    print(weekly_sales)


    def plot_with_trend(x_labels, y_values, title, xlabel):
        plt.figure(figsize=(12, 5))
        plt.plot(x_labels, y_values, marker='o', label='Sales')

        # Convert x to numeric indices for polyfit
        x_numeric = np.arange(len(x_labels))

        # Fit linear trend line (degree=1)
        coeffs = np.polyfit(x_numeric, y_values, deg=1)
        trend_line = coeffs[0] * x_numeric + coeffs[1]

        plt.plot(x_labels, trend_line, color='red', linestyle='--', label='Trend line')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.xticks(rotation=45)
        plt.ylabel('Total Sales Amount')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    # Plot monthly sales with trend
    plot_with_trend(monthly_sales['year_month'], monthly_sales['total_sales'],
                    'Monthly Sales Trend with Linear Trend Line', 'Year-Month')

    # Plot weekly sales with trend
    plot_with_trend(weekly_sales['year_week'], weekly_sales['total_sales'],
                    'Weekly Sales Trend with Linear Trend Line', 'Year-Week')

    con.close()
