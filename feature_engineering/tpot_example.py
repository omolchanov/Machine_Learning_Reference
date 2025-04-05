# https://machinelearningmastery.com/tpot-for-automated-machine-learning-in-python/

import warnings
import sys

warnings.filterwarnings('ignore')

from tpot import TPOTClassifier, TPOTRegressor

import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import get_scorer

from pyod.models.iforest import IForest

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(threshold=sys.maxsize, suppress=True)


def perform_classification():
    df = pd.read_csv('../assets/churn-rate.csv')

    X = df.drop(['state', 'area_code', 'phone_number', 'international_plan', 'voice_mail_plan'], axis=1)
    y = df['churn']

    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=1)
    model = TPOTClassifier(
        generations=2,
        population_size=10,
        cv=cv,
        scoring='accuracy',
        verbosity=2,
        random_state=1
    )

    model.fit(X, y)
    # model.export('clf_tpot_model.py')


def perform_regression():
    df_train = pd.read_csv('../assets/big_mart_sales_train.csv')
    df_test = pd.read_csv('../assets/big_mart_sales_test.csv')

    # Appending train and test dataframes
    df = df_train._append(df_test, ignore_index=True)

    # Removing unnecessary columns
    df = df.drop(['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year'], axis=1)

    # Dealing with missing values
    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
    df['Item_Outlet_Sales'].fillna(df['Item_Outlet_Sales'].mean(), inplace=True)
    df['Outlet_Size'].fillna('missing', inplace=True)

    # Features encoding and pre-processing
    num_columns = df.select_dtypes(include=np.number).columns
    cat_columns = df.select_dtypes(include='object').columns

    # Encoding 'Item_Fat_Content' feature to numerical
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
        'Low Fat': 0,
        'Regular': 1,
        'LF': 0,
        'reg': 1,
        'low fat': 0
    }, regex=True)

    num_pipeline = Pipeline(steps=[
        ('std', StandardScaler()),
        ('mm', MinMaxScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ('enc', OrdinalEncoder())
    ])

    ct = ColumnTransformer([
        ('num_pipeline', num_pipeline, num_columns),
        ('cat_pipeline', cat_pipeline, cat_columns)
    ])

    columns = num_columns.tolist() + cat_columns.tolist()
    df[columns] = ct.fit_transform(df)

    # Changing types after encoding
    df[cat_columns] = df[cat_columns].astype('int16')

    # Dealing with outliers
    model = IForest(n_estimators=1)
    model.fit(df)

    labels = model.labels_
    print(pd.Series(labels).value_counts())

    df = df[labels == 0]

    # Feature enginering
    df['Items_Outlet_Sold_Count'] = df['Item_Outlet_Sales'] / df['Item_MRP']
    df = df.replace([np.inf, -np.inf], 0)

    df = df.reset_index()

    # Training models and choosing the best one
    X = df.drop(['Item_Outlet_Sales'], axis=1)
    y = df['Item_Outlet_Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)
    model = TPOTRegressor(
        generations=3,
        population_size=10,
        scoring='r2',
        cv=cv,
        verbosity=2,
        random_state=1
    )

    model.fit(X_train, y_train)

    print('R2: %.16f' % (get_scorer('r2')(model, X_test, y_test)))
    print('nMAE: %.3f' % (abs(get_scorer('neg_mean_absolute_error')(model, X_test, y_test))))


if __name__ == '__main__':
    perform_classification()
    perform_regression()
