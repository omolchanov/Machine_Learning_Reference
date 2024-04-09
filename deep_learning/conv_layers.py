# Examples of convolutional layers for CNN
# Guideline:
# https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential
import keras.layers

import numpy as np


# Example of 1D Conv. layer (Vector)
def conv_layer_1d():
    data = np.asarray([0, 0, 0, 1, 1, 0, 0, 0])
    data = data.reshape(1, 8, 1)  # The input to Keras must be 3D for a 1D convolutional layer

    model = Sequential([
        keras.layers.Conv1D(1, 3, input_shape=(8, 1))
    ])

    # filter for detection of vertical lines
    weights = [
        np.asarray(
            [[[0]], [[1]], [[0]]]
        ),
        np.asarray([0.0])
    ]

    # Setting the filter (weights) to the model
    model.set_weights(weights)

    # Apply the filter to the input data. Calculating the feature map
    y_pred = model.predict(data)
    print(y_pred)


# Example of 2D Conv. layer (Matrix)
def conv_layer_2d():
    data = [
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0]
    ]

    data = np.asarray(data)
    data = data.reshape(1, 8, 8, 1)

    model = Sequential([
        keras.layers.Conv2D(1, (3, 3), input_shape=(8, 8, 1))
    ])

    # filter for detection of vertical lines
    detector = [[[[0]], [[1]], [[0]]],
                [[[0]], [[1]], [[0]]],
                [[[0]], [[1]], [[0]]]]

    weights = [np.asarray(detector), np.asarray([0.0])]
    model.set_weights(weights)

    y_pred = model.predict(data)

    for r in range(y_pred.shape[1]):
        # print each column in the row
        print([y_pred[0, r, c, 0] for c in range(y_pred.shape[2])])


conv_layer_1d()
conv_layer_2d()
