import duckdb
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
    df_clean = df.dropna(subset=['amount', 'churn', 'new_customer'])
    logging.info(f"Dataset shape after dropping NULL values: {df_clean.shape}")

    # Remove outliers
    logging.info(f"Dataset shape before removing outliers: {df_clean.shape}")

    Q1 = df_clean['amount'].quantile(0.25)
    Q3 = df_clean['amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_no_out = df_clean[(df_clean['amount'] >= lower_bound) & (df_clean['amount'] <= upper_bound)]
    logging.info(f"Dataset shape after removing outliers: {df_clean.shape}")

    return df_no_out


def func_train_model(data):
    df = data

    # Prepare features/target
    X = df[["amount", "new_customer"]]
    y = df["churn"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    logging.info(f"Model: {model}")
    logging.info(f"Model accuracy: {score:.3f}")


data = func_extract_data()
data = func_preprocess_data(data)
func_train_model(data)


