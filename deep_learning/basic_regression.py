# Guideline: https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
# Simple ANN for the regression problem implemented with Keras
# The legacy TF version is used in this script

import os
os.environ["TF_USE_LEGACY_KERAS"] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from scikeras.wrappers import KerasRegressor

from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
np.set_printoptions(suppress=True)


# Preparing the dataset
df = pd.read_csv('../assets/boston_houses.csv', header=None, sep='\s+')

X = df.iloc[:, 0:13]
y = df.iloc[:, 13]


def get_baseline_model():
    """
    Compiles a neural network model with input, hidden and output layers.
    :return: model
    """

    model = Sequential()

    model.add(Dense(13, input_shape=(13,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='relu', kernel_initializer='normal'))
    model.compile(loss='mean_absolute_error', optimizer='adam')

    return model


def get_larger_model():
    """
    Improves the performance of a neural network by adding more layers to the model.
    This might allow the model to extract and recombine higher-order features embedded in the data.

    :return: model
    """

    model = Sequential()

    model.add(Dense(13, input_shape=(13,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_absolute_error', optimizer='adam')

    return model


def get_wider_model():
    """
    Another approach to increasing the representational capability of the model is to create a wider network.
    The number of neurons in the hidden layer increased compared to the baseline model from 13 to 20.

    :return: model
    """

    model = Sequential()

    model.add(Dense(20, input_shape=(13,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_absolute_error', optimizer='adam')

    return model


def evaluate_non_processed_model(model):
    """
    Evaluates the model on the raw features' values (non-processed) from the dataframe
    :param model:
    """

    estimator = KerasRegressor(model=model, epochs=100, batch_size=50, verbose=False)
    kfold = KFold(n_splits=10)
    results = cross_val_score(estimator, X, y, cv=kfold, scoring='neg_mean_squared_error')

    print('Non Processed Data | MSE: %.2f | Std: %.2f' % (abs(results.mean()), results.std()))


def evaluate_std_model(model):
    """
    Evaluates the model on the pre-processed input data
    :param model:
    """

    cv = KFold(n_splits=10)

    estimators = [
        ('std', StandardScaler()),
        ('mlp', KerasRegressor(model=model, epochs=100, batch_size=50, verbose=False))
    ]

    pipeline = Pipeline(estimators)
    results = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_squared_error')

    print('Standardized Data | MSE: %.2f | Std: %.2f' % (abs(results.mean()), results.std()))


models = [
    ('Basic model', get_baseline_model()),
    ('Larger model', get_larger_model()),
    ('Wider model', get_wider_model())
]

for m in models:
    print('\n', m[0])
    evaluate_non_processed_model(m[1])
    evaluate_std_model(m[1])
