import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import models, layers, utils, backend as K
from Visualizer import Visualizer
import numpy as np


def present_model(model):
    # Summary and visualization
    model.summary()
    Visualizer.visualize_nn_structure(model, description=True, figsize=(6, 6))


# Perceptron
inputs = layers.Input(name="input", shape=(3,))
outputs = layers.Dense(name="output", units=1, activation='linear')(inputs)
model_perceptron = models.Model(inputs=inputs, outputs=outputs, name="Perceptron")


# DeepNN
# layer input
n_features=5
inputs = layers.Input(name="Input", shape=n_features)

# hidden layer 1
h1 = layers.Dense(name="Layer1", units=5, activation='relu')(inputs)

# hidden layer 2
h2 = layers.Dense(name="Layer2", units=5, activation='relu')(h1)

# layer output
outputs = layers.Dense(name="Output", units=1, activation='relu')(h2)

model_deep_nn = models.Model(inputs=inputs, outputs=outputs, name="Deep_NN_Keras")

present_model(model_perceptron)


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
model_perceptron.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',f1])

# Generate the dataset
X = np.random.rand(1000,3)
y = np.random.choice([1,0], size=1000)

# train/validation
training = model_perceptron.fit(x=X, y=y, batch_size=32, epochs=100, shuffle=False, verbose=1, validation_split=0.3)

# Visualising training results
Visualizer.visualize_training_results(training)
