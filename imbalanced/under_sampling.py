import warnings

from imblearn.under_sampling import (
    ClusterCentroids,
    RandomUnderSampler,
    NearMiss,
    TomekLinks,
    EditedNearestNeighbours,
    RepeatedEditedNearestNeighbours,
    AllKNN,
    CondensedNearestNeighbour,
    OneSidedSelection,
)
from imblearn.metrics import geometric_mean_score, specificity_score, sensitivity_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from collections import Counter

import pandas as pd

warnings.filterwarnings('ignore')

# Preparing the dataset
df = pd.read_csv('../assets/street_alert.csv')

days = {
    'monday': 1,
    'tuesday': 2,
    'wednesday': 3,
    'thursday': 4,
    'friday': 5,
    'saturday': 6,
    'sunday': 7
}

times_day = {'morning': 0, 'day': 1, 'evening': 2}

df['weekday'] = df['weekday'].replace(days)
df['time_day'] = df['time_day'].replace(times_day)

areas = df.groupby(['area']).size().sort_values(ascending=False).to_dict()
print('Original dataset:', '\n', areas)
print('N Classes: %.i' % len(areas.keys()))

# Filtering the dataset by number of appearance of a class
df = df.groupby('area').filter(lambda x: len(x) >= 0)
filtered_areas = df.groupby(['area']).size().sort_values(ascending=False).to_dict()
print('\n', 'Filtered dataset:', '\n', filtered_areas)
print('N Classes: %.i' % len(filtered_areas.keys()))

X = df[['weekday', 'time_day']]
y = df['area']


def evaluate_clf(X_resampled, y_resampled):
    clfs = [
        KNeighborsClassifier(n_neighbors=3),
        RandomForestClassifier(n_estimators=1000),
        LogisticRegression(max_iter=1000),
        SVC()
    ]

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)

    for clf in clfs:
        print('\n', clf)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print('Balanced accuracy: %.3f' % balanced_accuracy_score(y_test, y_pred))
        print('Geometric mean: %.3f' % geometric_mean_score(y_test, y_pred))
        print('Recall (sensitivity): %.3f' % sensitivity_score(y_test, y_pred, average='weighted'))
        print('Precision (specificity): %.3f' % specificity_score(y_test, y_pred, average='weighted'))


def sample(sampler):
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    a = sorted(Counter(y_resampled).items(), key=lambda x: (x[1], x[0]), reverse=True)
    print('\n', sampler)
    print('Number of classes: ', len(a))
    print(a)

    evaluate_clf(X_resampled, y_resampled)

    print('\n', '=======================')


samplers = [
    ClusterCentroids(random_state=0),
    RandomUnderSampler(),
    NearMiss(version=1, n_neighbors=1),
    TomekLinks(),
    EditedNearestNeighbours(),
    RepeatedEditedNearestNeighbours(),
    AllKNN(),
    CondensedNearestNeighbour(),
    OneSidedSelection(),
]

for s in samplers:
    sample(s)
