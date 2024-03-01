# Guildeline: https://www.youtube.com/watch?v=Zu5RzqdQJXo

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
from keras.datasets import cifar10
from keras.utils import to_categorical


input = keras.Input(shape=(32, 32, 3))

x = keras.layers.Conv2D(32, 3, activation='relu')(input)
x = keras.layers.MaxPooling2D(2, padding='same')(x)
x = keras.layers.Conv2D(64, 3, activation='relu')(x)
x = keras.layers.MaxPooling2D(2, padding='same')(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)

output = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(input, output)


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

print(model.evaluate(x_test, y_test))
