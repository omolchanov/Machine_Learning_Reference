import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import models, layers, backend as K
from Visualizer import Visualizer
import numpy as np

# DeepNN
# layer input
inputs = layers.Input(name="Input", shape=5)

# hidden layer 1
h1 = layers.Dense(name="Layer1", units=5, activation='relu')(inputs)

# hidden layer 2
h2 = layers.Dense(name="Layer2", units=5, activation='relu')(h1)

# layer output
outputs = layers.Dense(name="Output", units=1, activation='relu')(h2)

# Model object
model = models.Model(inputs=inputs, outputs=outputs, name="Deep_NN_Keras")

Visualizer.visualize_nn_structure(model)


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


# compile the neural network
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1])

# Generate the dataset for regression
# X = np.random.rand(1000, 5)
# y = np.random.uniform(low=0, high=0.2, size=1000)

# Generate the dataset for binary classification
X = np.random.rand(1000, 5)
y = np.random.choice([1, 0], size=1000)


# train/validation
training = model.fit(x=X, y=y, batch_size=32, epochs=100, shuffle=False, verbose=1, validation_split=0.3)

# Visualising training results
Visualizer.visualize_training_results(training)


def regression_prediction(model):
    x_test = np.random.rand(10, 5)
    y_pred = model.predict(x_test)

    for key, i in enumerate(y_pred):
        print("X_features: %s, Predicted result: %s" % (x_test[key], y_pred[key][0]))


def classification(model, threshold):
    x_test = np.random.rand(10, 5)
    y_pred = model.predict(x_test)

    for key, i in enumerate(y_pred):
        result = np.where(y_pred[key][0] > threshold, 1, 0)
        print("X_features: %s, Predicted probability: %s, Result: %s" % (x_test[key], y_pred[key][0], result))


classification(model, 0.5)
# regression_prediction(model_deep_nn)
