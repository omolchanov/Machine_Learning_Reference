import pandas as pd
import numpy as np

import pprint

from sktime.datasets import load_italy_power_demand
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

from sktime.transformations.series.summarize import SummaryTransformer
from sktime.transformations.series.exponent import ExponentTransformer

from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.feature_based import SummaryClassifier

from sktime.registry import all_estimators

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
pd.set_option("display.width", 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

np.set_printoptions(threshold=np.inf, suppress=True)

# Loading and splitting the dataset
X, y = load_italy_power_demand(return_type='numpy3D')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)


def basic_classification(clfs):
    for clf in clfs:
        print(clf)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred)

        print('Acc. score: %.3f' % acc_score)
        pprint.pprint(clf.get_tags())

        print('\n ================= \n')


def summary_transformer():
    tsf = SummaryTransformer()
    X_summaries = tsf.fit_transform(X)

    print(X_summaries)


def pipelines():
    pipe = ExponentTransformer() * KNeighborsTimeSeriesClassifier()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)

    print('Acc. score: %.3f' % acc_score)


def cross_validation(clfs):
    for clf in clfs:
        print(clf)

        cv = KFold(n_splits=5)
        cv_score = cross_val_score(clf, X_train, y_train, cv=cv)

        print('Mean accuracy: %.3f' % np.mean(cv_score))
        print('\n ================= \n')


def best_params():
    clf = KNeighborsTimeSeriesClassifier()

    param_grid = {'n_neighbors': range(2, 5 + 1), 'distance': ['euclidean', 'dtw']}

    cv = KFold(n_splits=2)
    gs = GridSearchCV(clf, param_grid, cv=cv)

    gs.fit(X_train, y_train)

    print('Best parameters: ', gs.best_params_)
    print('Best score: %.3f' % gs.best_score_)


def get_all_classifiers():
    print(all_estimators(
        'classifier',
        as_dataframe=True,
        filter_tags={'capability:multivariate': True}
    ))


classifiers = [KNeighborsTimeSeriesClassifier(), TimeSeriesForestClassifier(), SummaryClassifier()]

# basic_classification([
#     ,
#
# ])

# summary_transformer()
# pipelines()
# cross_validation(classifiers)
best_params()

# get_all_classifiers()
