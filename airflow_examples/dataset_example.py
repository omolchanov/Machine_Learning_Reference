from datetime import datetime

from airflow.decorators import dag, task
from airflow.datasets import Dataset

from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import classification_report

from tpot import TPOTClassifier

import pandas as pd


# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


CSV_DATASET_URI = '/opt/airflow/dags/df/churn-rate.csv'


@dag(
    dag_id='dataset_example',
    start_date=datetime(2025, 4, 15),
    schedule='@once',
    fail_stop=True,

)
def dataset_example_flow():

    @task(
        outlets=Dataset(CSV_DATASET_URI),
        show_return_value_in_logs=False
    )
    def download_df():
        try:
            df = pd.read_csv(CSV_DATASET_URI)
            print(df.shape)
            return df

        except FileNotFoundError:
            print('Can not download CSV dataframe')
            exit(1)

    @task(show_return_value_in_logs=False)
    def clear_df(df):
        df = df.drop(['area_code', 'phone_number'], axis=1)
        print(df.head())

        return df

    @task(show_return_value_in_logs=False)
    def encode_df(df):
        cat_columns = df[['state', 'international_plan', 'voice_mail_plan', 'churn']].columns
        cat_pipeline = Pipeline(steps=[
            ('enc', OrdinalEncoder())
        ])

        ct = ColumnTransformer([('cat_pipeline', cat_pipeline, cat_columns)])
        df[cat_columns] = ct.fit_transform(df)

        df[cat_columns] = df[cat_columns].astype('int16')

        print(df.info())
        # print(df.head())

        return df

    @task()
    def predict_churn(df):
        print(df.shape)

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

        y_pred = model.fitted_pipeline_.predict(X_test)
        print(classification_report(y_pred, y_test))

    df = download_df()
    df_cleared = clear_df(df)
    df_encoded = encode_df(df_cleared)
    predict_churn(df_encoded)


dataset_example_flow()
