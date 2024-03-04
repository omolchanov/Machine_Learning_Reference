# Guideline: https://machinelearningmastery.com/save-load-keras-deep-learning-models/

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Input

import pandas as pd

df = pd.read_csv('../assets/boston_houses.csv', header=None, sep='\s+')

X = df.iloc[:, 0:13]
y = df.iloc[:, 13]


def get_model() -> Sequential:

    model = Sequential([
        Input(shape=(13,)),
        Dense(13, activation='relu'),
        Dense(1, activation='relu')
    ])

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['r2_score'])
    model.fit(X, y, epochs=5, batch_size=15, verbose=True)

    return model


def save_model_to_json():

    # Saves model and its weights to JSON and H5 files
    model = get_model()

    model_json = model.to_json()
    with open('saved_models/model.json', 'w') as json_file:
        json_file.write(model_json)

    model.save_weights('saved_models/model.weights.h5')


def load_model_from_json() -> Sequential:

    # Load model from JSON file
    json_file = open('saved_models/model.json', 'r')
    model_json = json_file.read()
    json_file.close()

    model = model_from_json(model_json)

    # Load weights from H5 file
    model.load_weights('saved_models/model.weights.h5')

    # Model must be compiled after the loading
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['r2_score'])

    return model


def save_whole_model():

    # Saves the model and weights to .keras file
    model = get_model()
    model.save('saved_models/whole_model.keras')


def load_whole_model() -> Sequential:

    # Loads the model and weights from .keras file
    model = load_model('saved_models/whole_model.keras')
    return model


print('Saving/Loading the Model - JSON')
save_model_to_json()
model = load_model_from_json()

loss, accuracy = model.evaluate(X, y, verbose=True)
print('Loss (MAE): %.3f | R2: %.3f' % (loss, accuracy))

print('\nSaving/Loading the Model - Keras')
save_whole_model()
model = load_whole_model()

loss, accuracy = model.evaluate(X, y, verbose=True)
print('Loss (MAE): %.3f | R2: %.3f' % (loss, accuracy))
