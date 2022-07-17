import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
pd.set_option("display.precision", 2)
pd.set_option("expand_frame_repr", False)

from keras import models, layers, backend as K
from Visualizer import Visualizer
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# NN Configuration
NAME = 'Keras_Classification'
OPTIMIZER = 'adam'
LOSS = 'binary_crossentropy'
EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.3
THRESHOLD = 250


df = pd.read_csv('assets/churn-rate.csv')

numerical_columns = [
    'total day minutes',
    'total eve minutes',
    'total night minutes',
    'total intl minutes',
    'total day charge',
    'total eve charge',
    'total night charge',
    'total intl charge'
]

for i, name in enumerate(numerical_columns):
    df[name] = df[name].str.replace(",", ".").astype('float64')

X_minutes = [
    'total day minutes',
    'total eve minutes',
    'total night minutes',
    'total intl minutes'
]

X_calls = [
    'total day calls',
    'total eve calls',
    'total night calls',
    'total intl calls'
]

X_charges = [
    'total day charge',
    'total eve charge',
    'total night charge',
    'total intl charge'
]

y = df['churn'].map({False: 0, True: 1})

# Measure correlation between different sets of independent variables
# corr_minutes = df[X_minutes].copy().join(y).corr()
# corr_calls = df[X_calls].copy().join(y).corr()
corr_charges = df[X_charges].copy().join(y).corr()  # The strongest correlation between charges and churn

X = df[X_charges]

inputs = layers.Input(name="Input", shape=4)
h1 = layers.Dense(name="Layer1", units=4, activation='relu')(inputs)
# h2 = layers.Dense(name="Layer2", units=6, activation='relu')(h1)
outputs = layers.Dense(name="Output", units=1, activation='relu')(h1)

model = models.Model(inputs=inputs, outputs=outputs, name=NAME)

# model.summary()
# Visualizer.visualize_nn_structure(model)


# define metrics
def recall_func(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_func(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    precision = precision_func(y_true, y_pred)
    recall = recall_func(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy', f1])

X = np.array(X)
y = np.array(y)

training = model.fit(
    x=X,
    y=y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    shuffle=False,
    verbose=1,
    validation_split=VALIDATION_SPLIT
)

Visualizer.visualize_training_results(training)

x_pred = np.array([
    [265, 200, 245, 12],
    [365, 300, 345, 40],
    [1, 1, 1, 0],
    [130, 100, 100, 4]
])

y_pred = model.predict(x_pred)

for key, i in enumerate(y_pred):
    result = np.where(y_pred[key][0] > THRESHOLD, 1, 0)
    print("X_pred: %s, Predicted value: %s, Churn: %s" % (x_pred[key], y_pred[key][0], result))
