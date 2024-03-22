import sys

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold, cross_validate

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
np.set_printoptions(threshold=np.inf, suppress=True)
np.set_printoptions(threshold=sys.maxsize, suppress=True)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)

df = pd.read_csv('../assets/winequality-white.csv', sep=';')


def build_whiskey_plot():
    """
    Builds a Whiskey plot for easier identification of outliers
    """

    fig, ax = plt.subplots(1, df.shape[1], sharex=False)

    for i, c in enumerate(df.columns):
        sns.boxplot(y=df[c], ax=ax[i])

    plt.subplots_adjust(wspace=1.75)
    plt.show()


def print_result(func):
    """
    Decorator for various pre-processing functions for printing the dataframe's shape
    :param func: pre-processing function
    :return: wrapper
    """

    def wrapper(*args):
        func(*args)
        print('\n', func.__name__, df.shape)
    return wrapper


@print_result
def remove_outliers():
    """
    Removes the outliers from all the features basing InterQuartile Range
    """

    outlier_columns = df.columns[:-2]

    for c in outlier_columns:

        q1 = np.percentile(df[c], 25)
        q3 = np.percentile(df[c], 75)

        iqr = q3 - q1

        upper = np.where(df[c] >= (q3 + 0.4 * iqr))
        lower = np.where(df[c] <= (q3 - 1.5 * iqr))

        df.drop(upper[0], inplace=True)
        df.drop(lower[0], inplace=True)

        df.reset_index(drop=True, inplace=True)


@print_result
def remove_unimportant_features():
    """
    Removes the unimportant features basing on Descision Tree calculation of importances
    """

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    importances = clf.feature_importances_.argsort()[::-1]

    # for i in importances:
    #     print(df.columns[i])

    df.drop(df.columns[importances[-4:]], axis=1, inplace=True)


@print_result
def drop_minority_classes(threshold=0):
    """
    Drops the minority classes basing on the threshold of the class' instances
    :param threshold: int, default: 0
    """

    idx = df.groupby('quality').filter(lambda x: len(x) < threshold).index
    df.drop(idx, inplace=True)

    df.reset_index(drop=True, inplace=True)

    # print('\n The distribution of the classes: ', df['quality'].value_counts())


@print_result
def remove_low_correlated_features():
    """
    Removes the unimportant features basing on correlation with the target variable
    """

    corr = df.corr()['quality'].sort_values(ascending=False)
    print(corr)

    df.drop([
        'citric acid',
        'residual sugar',
        'fixed acidity',
        'total sulfur dioxide',
        'volatile acidity',
        'chlorides',
        'density'
    ], axis=1, inplace=True)


def preprocess_dataframe():
    """
    Preprocess the independent variable with different scalers
    :return:
        X_pr - a matrix with pre-processed independent variable
        y - vector with dependent variable
    """

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].values

    pipe = Pipeline([
        ('x_std', StandardScaler()),
        ('min_max', MinMaxScaler())
    ])

    X_pr = pipe.fit_transform(X)

    return X_pr, y


if __name__ == '__main__':

    drop_minority_classes(500)
    remove_unimportant_features()
    # remove_low_correlated_features()

    remove_outliers()
    build_whiskey_plot()

    print('\n', df.describe())

    X, y = preprocess_dataframe()

    # Oversampling
    X, y = SMOTE().fit_resample(X, y)


    def cross_validation(model):
        """
        Cross-validates the model
        :param model
        :return: scores
        """

        k = 5
        cv = KFold(n_splits=k)

        scores = cross_validate(model, X, y, cv=cv, scoring=['accuracy', 'neg_mean_absolute_error'])
        return scores


    models = (
        LogisticRegression(max_iter=10000),
        SVC(C=10),
    )

    for m in models:
        print('\n', m)
        print(X.shape)

        mean_acc = np.mean(cross_validation(m)['test_accuracy'])
        mean_loss = np.mean(cross_validation(m)['test_neg_mean_absolute_error'])

        print('Mean accuracy: %.3f' % mean_acc)
        print('Mean loss: %.3f' % np.abs(mean_loss))
