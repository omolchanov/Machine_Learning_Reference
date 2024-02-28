# Guideline: https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
# Simple ANN for the regression problem implemented with Keras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

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


def get_model(size):
    """
    Returns a NN model of a particular configuration
    :param size: type of the model [basic, large, wide]
    :return: model's name, model
    """

    model = Sequential()

    # Compiles a neural network model with input, hidden and output layers
    if size == 'basic':

        # Initializers define the way to set the initial random weights of Keras layers
        # https://keras.io/api/layers/initializers/
        model.add(Dense(1, activation='relu', kernel_initializer='normal'))

        model.name = 'Basic_model'

    # Improves the performance of a neural network by adding more layers to the model.
    # This might allow the model to extract and recombine higher-order features embedded in the data
    if size == 'large':
        model.add(Dense(13, kernel_initializer='normal', activation='relu'))
        model.add(Dense(13, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))

        model.name = 'Large_model'

    # Another approach to increasing the representational capability of the model is to create a wider network.
    # The number of neurons in the hidden layer increased compared to the baseline model from 13 to 20
    if size == 'wide':
        model.add(Dense(20, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))

        model.name = 'Wide_model'

    return model


def evaluate(model, preprocess_data=False):
    """
    Evaluates the model with non-processed and processed (std) data
    :param preprocess_data: boolean. If true, X is pre-processed with StandardScaler()
    :param model:
    """

    X_data = X
    data_str = 'Non-processed data'

    if preprocess_data is True:
        std_scaler = StandardScaler()
        X_scaled = std_scaler.fit_transform(X)

        X_data = X_scaled
        data_str = 'Pre-processed (std) data'

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['r2_score'])
    model.fit(X_data, y, epochs=150, batch_size=15, verbose=False)

    loss, _ = model.evaluate(X, y)
    print('%s | MAE: %.3f' % (data_str, loss))


models = [
    get_model(size='basic'),
    get_model(size='large'),
    get_model(size='wide')
]

for m in models:
    print('\n', m.name)
    evaluate(m)
    evaluate(m, preprocess_data=True)
