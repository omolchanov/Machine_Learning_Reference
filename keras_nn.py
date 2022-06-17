from keras import models, layers, utils, backend as K
from Visualizer import visualize_nn


def present_model(model):
    # Summary and visualization
    model.summary()
    visualize_nn(model, description=True, figsize=(6, 6))


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

present_model(model_deep_nn)
