# https://www.datacamp.com/tutorial/introduction-to-anomaly-detection
# https://www.statisticshowto.com/median-absolute-deviation/
# https://medium.com/@corymaklin/isolation-forest-799fceacdda4

import sys

import pandas as pd
import numpy as np

from pyod.models.mad import MAD
from pyod.models.iforest import IForest

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
np.set_printoptions(threshold=sys.maxsize, suppress=True)

df = pd.read_csv('../assets/churn-rate.csv')
# print(df.columns)


def remove_unvariate_outliers():
    """
    Univariate outliers exist in a single variable or feature in isolation. Univariate outliers are extreme or
    abnormal values that deviate from the typical range of values for that specific feature.
    """

    X = df[['customer_service_calls']]

    print('===== BEFORE REMOVING OUTLIERS ======')
    print(df.shape, '\n')
    print(X.value_counts(), '\n')

    model = MAD()
    model.fit(X)

    labels = model.labels_
    print(pd.Series(labels).value_counts())
    print('\nOutliers index: ', df[labels == 1].index)

    print('\n===== AFTER REMOVING OUTLIERS ======')
    new_df = df[labels == 0]
    print(new_df.shape)


def remove_multivariate_outliers():
    """
    Multivariate outliers are found by combining the values of multiple variables at the same time.
    """

    X = df.drop([
        'state',
        'area_code',
        'phone_number',
        'international_plan',
        'voice_mail_plan',
        'churn'
    ], axis=1)

    print('\n===== BEFORE REMOVING OUTLIERS ======')
    print(df.shape, '\n')

    model = IForest(n_estimators=1000)
    model.fit(X)

    labels = model.labels_
    print(pd.Series(labels).value_counts())

    print('\nOutliers index: ', df[labels == 1].index.to_numpy())

    print('\n===== AFTER REMOVING OUTLIERS ======')
    new_df = df[labels == 0]
    print(new_df.shape)


if __name__ == '__main__':
    remove_unvariate_outliers()
    remove_multivariate_outliers()
