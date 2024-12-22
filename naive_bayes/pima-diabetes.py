# https://habr.com/ru/articles/739648/

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd

df = pd.read_csv('../assets/pima-diabetes.csv', header=None)

# Converting all types to float
df = df.astype(float)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Splitting onto train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predicting with different NB classifiers
clfs = [
    GaussianNB(),
    MultinomialNB(),
    BernoulliNB()
]

for i, clf in enumerate(clfs):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('%s Accuracy score: %.2f' % (clf, accuracy_score(y_pred, y_test)))
    print(classification_report(y_pred, y_test))
