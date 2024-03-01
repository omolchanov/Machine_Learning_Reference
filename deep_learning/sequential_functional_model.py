# Guideline: https://www.youtube.com/watch?v=6622Py-mymY

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
np.set_printoptions(threshold=sys.maxsize, suppress=True)

import tensorflow as tf
import pandas as pd

from keras.models import Sequential, Model
from keras.layers import Dense, Input


from sklearn.model_selection import train_test_split

# Preparing the dataset
df = pd.read_csv('../assets/boston_houses.csv', header=None, sep='\s+')

X = df.iloc[:, 0:13]
y = df.iloc[:, 13]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)


# Sequential model
model = Sequential([
    Input(shape=(13,)),
    Dense(10, activation='relu', name='hidden_1'),
    Dense(2, activation='softmax', name='output')
])

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
model.fit(X_train, y_train, epochs=5, batch_size=15, verbose=True)

# Functional model
model_r = Model(model.inputs, model.get_layer(name='output').output)

x = tf.convert_to_tensor([X.iloc[0].values])
y1 = model(x)
y2 = model_r(x)

# Extended Sequential model
model_ex = Sequential([
    model,
    Dense(30, activation='tanh')
])

model.trainable = False

model_ex.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
model_ex.fit(X_train, y_train, epochs=3, batch_size=15, verbose=True)
