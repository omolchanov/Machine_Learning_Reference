# https://machinelearningmastery.com/tpot-for-automated-machine-learning-in-python/

import warnings
warnings.filterwarnings('ignore')

from tpot import TPOTClassifier, TPOTRegressor
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.preprocessing import LabelEncoder

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def perform_classification():
    df = pd.read_csv('../assets/churn-rate.csv')

    X = df.drop(['state', 'area_code', 'phone_number', 'international_plan','voice_mail_plan'], axis=1)
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

    # Dealing with missing values
    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
    df['Item_Outlet_Sales'].fillna(df['Item_Outlet_Sales'].mean(), inplace=True)
    df['Outlet_Size'].fillna('missing', inplace=True)

    df = df.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'], axis=1)

    # Encoding 'Item_Fat_Content' feature to numerical
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
        'Low Fat': 0,
        'Regular': 1,
        'LF': 0,
        'reg': 1,
        'low fat': 0
    }, regex=True)

    # Encoding categorical features to numerical
    # df = df.apply(LabelEncoder().fit_transform)
    df['Item_Type'] = LabelEncoder().fit_transform(df['Item_Type'])
    df['Outlet_Size'] = LabelEncoder().fit_transform(df['Outlet_Size'])
    df['Outlet_Location_Type'] = LabelEncoder().fit_transform(df['Outlet_Location_Type'])
    df['Outlet_Type'] = LabelEncoder().fit_transform(df['Outlet_Type'])

    X = df.drop('Item_MRP', axis=1)
    y = df['Item_MRP']

    cv = RepeatedKFold(n_splits=2, n_repeats=1, random_state=1)

    model = TPOTRegressor(
        generations=1,
        population_size=10,
        scoring='neg_mean_absolute_error',
        cv=cv,
        verbosity=2,
        random_state=1
    )

    model.fit(X, y)


if __name__ == '__main__':
    perform_classification()
    perform_regression()
