import sys

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from pyod.models.abod import ABOD
from pyod.models.iforest import IForest

import pandas as pd
import numpy as np

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
np.set_printoptions(threshold=sys.maxsize, suppress=True)

df = pd.read_csv('../assets/cc_dataset.csv', index_col='CUST_ID')
df = df.dropna()


def remove_outliers(od):
    """
    Drops outliers with chosen PYOD outlier remover
    :param od: instance of PYOD model
    :return: dataframe without outliers
    """

    print('Removing outliers with', od)
    print('DF shape: ', df.shape)

    X = df.iloc[:, :-1]
    od.fit(X)

    labels = od.labels_
    print('Outliers statistic:\n', pd.Series(labels).value_counts())

    df_wo = df[labels == 0]
    print('DF shape without outliers: ', df_wo.shape)

    return df_wo


def predict(reg, df):
    """
    Predicts the target variable with chosen regressor. Evaluates the prediction's results
    :param reg: instance of Sklearn regressor model
    :param df: dataframe
    """
    X = df.iloc[:, :-1]
    y = df['TENURE']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

    print('\nPredicting with', reg)
    print('DF shape: ', df.shape)

    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    print('MAE: %.3f' % (mean_absolute_error(y_pred, y_test)))
    print('R2: %.3f' % (r2_score(y_pred, y_test)))
    print('==============\n')


def clean_and_predict(od, reg):
    predict(reg, remove_outliers(od))


def run_service():
    regs = [
        SVR(),
        DecisionTreeRegressor(),
        Ridge(),
        Lasso(),
        KNeighborsRegressor()
    ]

    # Predicting without removing outliers
    for _, r in enumerate(regs):
        predict(r, df)

    ods = [
        IForest(),
        ABOD(),
    ]

    # Predicting with removing outliers
    for _, od in enumerate(ods):
        for _, reg in enumerate(regs):
            clean_and_predict(od, reg)


if __name__ == '__main__':
    run_service()
