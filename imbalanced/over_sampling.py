# Guideline https://imbalanced-learn.org/stable/over_sampling.html

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.decomposition import PCA

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from collections import Counter

import pandas as pd


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

df = df.groupby('area').filter(lambda x: len(x) >= 4)

X = df[['weekday', 'time_day']]
y = df['area']

pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)

areas = df.groupby(['area']).size().sort_values(ascending=False).to_dict()
print('Original dataset:', '\n', areas)


def naive_random_over_sampling():
    res = RandomOverSampler()
    X_resampled, y_resampled = res.fit_resample(X, y)

    print('\n', res)
    print(sorted(Counter(y_resampled).items()))

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)

    clfs = [
        KNeighborsClassifier(n_neighbors=1),
        RandomForestClassifier(n_estimators=500),
        LogisticRegression(max_iter=1000)
    ]

    for clf in clfs:
        print('\n', clf)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
        print('Precision: %.3f' % precision_score(y_test, y_pred, average='weighted', zero_division=True))
        print('Recall: %.3f' % recall_score(y_test, y_pred, average='weighted', zero_division=True))


def smote_adasyn_sampling():
    clfs = [
        KNeighborsClassifier(n_neighbors=1),
        RandomForestClassifier(n_estimators=1000),
        LogisticRegression(max_iter=1000)
    ]

    # Sampling with Synthetic Minority Oversampling Technique (SMOTE) method
    sam = SMOTE(k_neighbors=3)
    X_resampled, y_resampled = sam.fit_resample(X, y)

    a = sorted(Counter(y_resampled).items(), key=lambda x: (x[1], x[0]), reverse=True)
    print('\n', sam, ':', a)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)

    for clf in clfs:
        print('\n', clf)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
        print('Precision: %.3f' % precision_score(y_test, y_pred, average='weighted', zero_division=True))
        print('Recall: %.3f' % recall_score(y_test, y_pred, average='weighted', zero_division=True))

    # Sampling with Adaptive Synthetic (ADASYN) method
    sam = ADASYN(n_neighbors=2, sampling_strategy='minority')
    X_resampled, y_resampled = sam.fit_resample(X, y)

    a = sorted(Counter(y_resampled).items(), key=lambda x: (x[1], x[0]), reverse=True)
    print('\n', sam, ':', a)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)

    for clf in clfs:
        print('\n', clf)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
        print('Precision: %.3f' % precision_score(y_test, y_pred, average='weighted', zero_division=True))
        print('Recall: %.3f' % recall_score(y_test, y_pred, average='weighted', zero_division=True))


naive_random_over_sampling()
smote_adasyn_sampling()
