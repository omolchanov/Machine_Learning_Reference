import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import datetime
import tensorflow as tf
from tensorflow.keras import layers, models

import numpy as np


model = models.Sequential([
    layers.Input(shape=(8,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])


x_train = np.random.rand(500, 8)
y_train = np.random.randint(0, 2, size=(500, 1))


# TensorBoard log setup
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True
)

# Train with callback
model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    callbacks=[tensorboard_callback]
)

print("\n✅ Training complete. Now run this in your terminal:")
print(f"tensorboard --logdir logs/fit")
