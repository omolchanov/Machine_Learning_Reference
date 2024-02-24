# Basic neural network with Keras
# Guideline: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

import warnings
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputSpec

warnings.filterwarnings('ignore')

# Configuration
np.set_printoptions(threshold=np.inf, suppress=True)

df = np.loadtxt('../assets/pima-diabetes.csv', delimiter=',')
X = df[:, 0:8]
y = df[:, 8]


model = Sequential()

model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=150, batch_size=10, verbose=0)

# Evaluate the accuracy
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

pred = (model.predict(X) > 0.5).astype(int)
print(pred)
