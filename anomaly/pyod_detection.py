# https://www.datacamp.com/tutorial/introduction-to-anomaly-detection
# https://www.statisticshowto.com/median-absolute-deviation/
# https://medium.com/@corymaklin/isolation-forest-799fceacdda4

import sys

import pandas as pd
import numpy as np

from pyod.models.mad import MAD
from pyod.models.iforest import IForest
from pyod.models.pca import PCA
from pyod.utils.data import generate_data, get_outliers_inliers, evaluate_print
from pyod.utils.example import visualize

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns


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


def remove_outliers_pca():
    X_train, y_train = generate_data(train_only=True)

    df_train = pd.DataFrame(X_train)
    df_train['y'] = y_train

    clf = PCA()
    clf.fit(X_train)

    # Binary labels of the training data, where 0 indicates inliers and 1 indicates outliers/anomalies.
    y_train_pred = clf.labels_
    print(pd.Series(y_train_pred).value_counts())

    # Outlier scores of the training data. Higher scores typically indicate more abnormal behavior.
    # Outliers usually have higher scores. Outliers tend to have higher scores.
    y_train_scores = clf.decision_scores_
    y_train_scores_scaled = MinMaxScaler().fit_transform(y_train_scores.reshape(len(y_train_scores), 1))

    df_s = pd.DataFrame({
        'x0': df_train[0],
        'x1': df_train[1],
        'y': y_train_scores_scaled[:,0]

    })

    ax = sns.scatterplot(x='x0', y='x1', hue='y', data=df_s, palette="RdBu_r")
    ax.legend(title="Anomaly Scores")
    plt.show()


if __name__ == '__main__':
    remove_unvariate_outliers()
    remove_multivariate_outliers()
    remove_outliers_pca()
