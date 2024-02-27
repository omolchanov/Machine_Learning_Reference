# A basic neural network for binary classification with Keras
# Guideline: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

# Supressing the TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Configuration
np.set_printoptions(threshold=np.inf, suppress=True)

# Dataset
df = np.loadtxt('../assets/pima-diabetes.csv', delimiter=',')
X = df[:, 0:8]
y = df[:, 8]

model = Sequential()

# https://keras.io/api/layers/core_layers/dense/
# https://robotdreams.cc/blog/327-funkciji-aktivaciji-stupinchasta-liniyna-sigmojida-relu-ta-tanh#item6
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# https://keras.io/api/losses/probabilistic_losses/#binarycrossentropy-class
# https://analyticsindiamag.com/guide-to-tensorflow-keras-optimizers/
# https://towardsdatascience.com/stochastic-gradient-descent-clearly-explained-53d239905d31
# https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# https://neurohive.io/ru/osnovy-data-science/jepoha-razmer-batcha-iteracija/
model.fit(X, y, epochs=150, batch_size=10, verbose=True)

# Evaluate the accuracy and loss
loss, accuracy = model.evaluate(X, y)
print('Loss: %.2f | Accuracy: %.2f' % (loss, accuracy*100))

# Making predictions
pred = (model.predict(X) > 0.5).astype(int)
