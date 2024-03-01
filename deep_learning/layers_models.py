import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pprint

from keras.layers import Input, Dense
from keras.models import Sequential, Model

import pandas as pd
import numpy as np

df = pd.read_csv('../assets/boston_houses.csv', header=None, sep='\s+').head(10)

X = df.iloc[:, 0:13]
y = df.iloc[:, 13]


def nn_input_layer():
    """
    A simple NN with the Input, Dense and Output layers.
    Functional API is used
    """

    a = Input(shape=(13, ))
    b = Dense(20, activation='relu')(a)
    c = Dense(1, activation='relu')(b)

    model = Model(a, c)

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['r2_score'])
    model.fit(X, y, epochs=1, batch_size=5, verbose=False)

    loss, r2 = model.evaluate(X, y)
    print('Loss: %.3f | R2: %.3f' % (loss, r2))

    pprint.pp(b.output)


def seq_model():
    model = Sequential([
        Input(shape=(13,)),
        Dense(13, activation='relu', use_bias=False)
    ])

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['r2_score'])
    model.fit(X, y, epochs=1, batch_size=5, verbose=False)

    print(model.summary(expand_nested=True))

nn_input_layer()
seq_model()
