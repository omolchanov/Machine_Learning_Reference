# Cross validation of a Keras model
# Guildeine: https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-keras.md

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import KFold

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt


# Preparing the dataset
df = pd.read_csv('../assets/boston_houses.csv', header=None, sep='\s+')

X = df.iloc[:, 0:13]
y = df.iloc[:, 13]

k = 12
cv = KFold(n_splits=k)

test_scores, train_scores, folds = [], [], []


def get_model():
    model = Sequential()
    model.add(Dense(1, activation='relu', kernel_initializer='normal'))

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    model.fit(X, y, epochs=150, batch_size=15, verbose=False)

    return model


f = 1
for train, test in cv.split(X, y):
    model = get_model()

    test_score = model.evaluate(np.take(X, test, axis=0), np.take(y, test, axis=0), verbose=0)
    train_score = model.evaluate(np.take(X, train, axis=0), np.take(y, train, axis=0), verbose=0)

    test_scores.append(test_score[1])
    train_scores.append(train_score[1])
    folds.append(f)

    f = f+1

plt.plot(folds, test_scores)
plt.plot(folds, train_scores)

plt.title('Cross validation of NN model')
plt.legend(['Test scores', 'Train scores'])
plt.xlabel('Fold')
plt.ylabel('MAE, K$')

plt.show()
