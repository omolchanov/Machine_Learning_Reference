import warnings

from imblearn.ensemble import (
    BalancedBaggingClassifier,
    BalancedRandomForestClassifier,
    RUSBoostClassifier,
    EasyEnsembleClassifier
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

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
df = df.groupby('area').filter(lambda x: len(x) >= 16)
filtered_areas = df.groupby(['area']).size().sort_values(ascending=False).to_dict()
print('\n', 'Filtered dataset:', '\n', filtered_areas)
print('N Classes: %.i' % len(filtered_areas.keys()))

X = df[['weekday', 'time_day']]
y = df['area']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


def evaluate(clf):
    print('\n', clf)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print('Accuracy: %.3f' % balanced_accuracy_score(y_test, y_pred))


classifiers = [
    BalancedBaggingClassifier(RandomForestClassifier(n_estimators=1000)),
    BalancedRandomForestClassifier(n_estimators=100, sampling_strategy='all', replacement=True, bootstrap=False),
    RUSBoostClassifier(n_estimators=200, algorithm='SAMME.R', random_state=0),
    EasyEnsembleClassifier(random_state=0)
]

for clf in classifiers:
    evaluate(clf)
