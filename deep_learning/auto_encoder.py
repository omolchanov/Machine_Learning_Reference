# Guideline:
# https://www.geeksforgeeks.org/auto-encoders/

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import layers
from keras.datasets import mnist
from keras.models import Model, Sequential

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

# Loading the MNIST dataset
(X_train, _), (X_test, _) = mnist.load_data()

# Normalizing pixel values to the range [0, 1]
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.


class Autoencoder(Model):

    def __init__(self, latent_dimensions: int, data_shape) -> None:
        super().__init__()

        self.latent_dimensions = latent_dimensions
        self.data_shape = data_shape

        # Encoder architecture using a Sequential model
        self.encoder = Sequential([
            layers.Flatten(),
            layers.Dense(latent_dimensions, activation='relu')
        ])

        # Decoder architecture using another Sequential model
        self.decoder = Sequential([

            # https://www.geeksforgeeks.org/python-tensorflow-math-reduce_prod/
            layers.Dense(tf.math.reduce_prod(data_shape), activation='sigmoid'),

            # https://keras.io/api/layers/reshaping_layers/reshape/
            layers.Reshape(data_shape)
        ])

    def call(self, input_data, **kwargs):
        encoded_data = self.encoder(input_data)
        decoded_data = self.decoder(encoded_data)

        return decoded_data


if __name__ == '__main__':
    # Extracting shape information from the testing dataset
    input_data_shape = X_test.shape[1:]

    # Specifying the dimensionality of the latent space
    latent_dimensions = 64

    auto_enc = Autoencoder(latent_dimensions, input_data_shape)

    auto_enc.compile(optimizer='adam', loss='mean_squared_error')
    auto_enc.fit(X_train, X_train, epochs=1, shuffle=True, validation_data=(X_test, X_test))

    encoded_imgs = np.array(auto_enc.encoder(X_test))
    decoded_imgs = auto_enc.decoder(encoded_imgs).numpy()

    # Visualising original and reconstructed data
    plt.figure(figsize=(8, 4))

    n = 10
    for i in range(n):

        # Original data
        plt.subplot(2, n, i + 1)

        # https://www.geeksforgeeks.org/matplotlib-pyplot-imshow-in-python/
        plt.imshow(X_test[i])
        plt.title('Original')

        # Reconstructed data
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.title('Reconstructed')

    plt.show()
