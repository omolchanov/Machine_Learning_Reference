import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import models, layers, backend as K
from Visualizer import Visualizer

# Perceptron
inputs = layers.Input(name="input", shape=(3,))
outputs = layers.Dense(name="output", units=1, activation='linear')(inputs)

# Model object
model = models.Model(inputs=inputs, outputs=outputs, name="Perceptron")

model.summary()
Visualizer.visualize_nn_structure(model)
