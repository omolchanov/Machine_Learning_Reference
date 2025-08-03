import duckdb
import pandas as pd

import logging

logging.basicConfig(level=logging.INFO)


def func_extract_data():
    # df = pd.read_csv('data/stagins_sales_df.csv')
    df = pd.read_csv('/opt/airflow/dags/data/stagins_sales_df.csv')

    return df


def func_preprocess_data(data):
    df = data
    logging.info(f"Dataset shape before processing: {df.shape}")

    # Drop rows with missing 'amount'
    df_clean = df.dropna(subset=['amount'])
    logging.info(f"Dataset shape after dropping NULL values: {df_clean.shape}")

    # Remove outliers
    logging.info(f"Dataset shape before removing outliers: {df_clean.shape}")

    Q1 = df_clean['amount'].quantile(0.25)
    Q3 = df_clean['amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_clean = df_clean[(df_clean['amount'] >= lower_bound) & (df_clean['amount'] <= upper_bound)]
    logging.info(f"Dataset shape after removing outliers: {df_clean.shape}")


data = func_extract_data()
func_preprocess_data(data)


