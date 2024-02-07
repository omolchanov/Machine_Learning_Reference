import pandas as pd
import numpy as np

import pprint


from sktime.datasets import load_italy_power_demand
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.interval_based import TimeSeriesForestClassifier

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


basic_classification([
    KNeighborsTimeSeriesClassifier(),
    TimeSeriesForestClassifier()
])
