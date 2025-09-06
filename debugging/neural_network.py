import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import datetime
import tensorflow as tf
from tensorflow.keras import layers, models

import numpy as np

np.set_printoptions(threshold=np.inf)


class PrintAllWeights(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\n=== Epoch {epoch+1} Weights ===")

        for i, layer in enumerate(self.model.layers):
            weights = layer.get_weights()

            if weights:  # only layers with weights
                kernel, bias = weights

                print("\n==================")

                print(f"\nLayer {i} - {layer.name}")
                print("Kernel (weights between neurons):")
                print(kernel)   # full weight matrix

                print("Biases (one per output neuron):")
                print(bias)     # bias vector


class WeightsToTensorBoard(tf.keras.callbacks.Callback):
    def __init__(self, log_dir="./logs/weights"):
        super().__init__()
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        with self.writer.as_default():
            for layer in self.model.layers:
                for weight in layer.weights:
                    # Log full weight tensor as histogram
                    tf.summary.histogram(
                        name=weight.name,
                        data=weight,
                        step=epoch
                    )
        self.writer.flush()


model = models.Sequential([
    layers.Input(shape=(8,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


x_train = np.random.rand(15000, 8)
y_train = np.random.randint(0, 2, size=(15000, 1))


# TensorBoard log setup
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True
)
weights_cb = WeightsToTensorBoard(log_dir=log_dir + "/weights")

# Train with callback
model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    callbacks=[tensorboard_callback, weights_cb, PrintAllWeights()],
    verbose=2
)

print("\nTraining complete. Now run this in your terminal:")
print(f"tensorboard --logdir logs/fit")

model.save(f"{log_dir}/my_model.h5")
