import io
import requests

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.datasets import Dataset

from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import get_scorer

from tpot import TPOTClassifier

import pandas as pd
import numpy as np


# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


csv_dataset = Dataset(
    'https://upnow-prod.ff45e40d1a1c8f7e7de4e976d0c9e555.r2.cloudflarestorage.com/QVVjE28OTFfPalQb6D5tOAXjWiy2/a78cb711'
    '-2e80-4cf4-af91-4b0e5ed16705?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=cdd12e35bbd220303957dc5603a4cc8e%2F'
    '20250417%2Fauto%2Fs3%2Faws4_request&X-Amz-Date=20250417T212625Z&X-Amz-Expires=43200&X-Amz-Signature=e23e3fa362ec89'
    '23ca395a9d629b2a5a168df2e2c1e336d1507f827b9f8e2657&X-Amz-SignedHeaders=host&response-content-disposition='
    'attachment%3B%20filename%3D%22churn-rate.csv%22')


def download_df(**kwargs):
    try:
        url = csv_dataset.uri
        s = requests.get(url).content
        df = pd.read_csv(io.StringIO(s.decode('utf-8')))

        kwargs['ti'].xcom_push(value=df, key='df')

    except FileNotFoundError:
        print('Can not download CSV dataframe')


def clear_df(**kwargs):
    df:pd.DataFrame = kwargs['ti'].xcom_pull(task_ids='df_download_task', key='df')
    df = df.drop(['area_code', 'phone_number'], axis=1)

    print(df.head())

    kwargs['ti'].xcom_push(value=df, key='df')


def encode_df(**kwargs):
    df: pd.DataFrame = kwargs['ti'].xcom_pull(task_ids='clear_df_task', key='df')

    cat_columns = df[['state', 'international_plan', 'voice_mail_plan', 'churn']].columns
    cat_pipeline = Pipeline(steps=[
        ('enc', OrdinalEncoder())
    ])

    ct = ColumnTransformer([('cat_pipeline', cat_pipeline, cat_columns)])
    df[cat_columns] = ct.fit_transform(df)

    df[cat_columns] = df[cat_columns].astype('int16')

    print(df.info())
    print(df.head())

    kwargs['ti'].xcom_push(value=df, key='df')


def predict_churn(**kwargs):
    df: pd.DataFrame = kwargs['ti'].xcom_pull(task_ids='encode_df_task', key='df')
    print(df.head())

    X = df.iloc[:, :-1]
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)

    model = TPOTClassifier(
        generations=1,
        population_size=10,
        cv=cv,
        scoring='accuracy',
        verbosity=2,
        random_state=1
    )

    model.fit(X_train, y_train)
    model.export('clf_tpot_model.py')

    print('Accuracy: %.3f' % (get_scorer('accuracy')(model, X_test, y_test)))


with DAG(
    dag_id='dataset_example',
    start_date=datetime(2025, 4, 15),
    schedule='@once',
    fail_stop=True

) as dag:

    download_df_task = PythonOperator(
        task_id='df_download_task',
        outlets=csv_dataset,
        python_callable=download_df,
    )

    clear_df_task = PythonOperator(
        task_id='clear_df_task',
        python_callable=clear_df
    )

    encode_df_task = PythonOperator(
        task_id='encode_df_task',
        python_callable=encode_df
    )

    predict_churn_task = PythonOperator(
        task_id='predict_churn_task',
        python_callable=predict_churn
    )

    download_df_task >> clear_df_task >> encode_df_task >> predict_churn_task
