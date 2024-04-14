# Guidelines:
# https://towardsdatascience.com/visualizing-backpropagation-in-neural-network-training-2647f5977fdb
# https://facebookresearch.github.io/hiplot/

# To start the HiPlot server run "python -m hiplot"

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import layers
from keras import models
from keras.datasets import boston_housing

from sklearn.preprocessing import StandardScaler

import hiplot as hip
import numpy as np

# Configuration
np.set_printoptions(threshold=sys.maxsize, suppress=True)

# Loading the dataset
df = boston_housing.load_data()

# Splitting onto train and test holdouts
(x_train, y_train), (x_test, y_test) = df

# Normalizing data
std_sc = StandardScaler()
x_train = std_sc.fit_transform(x_train)
x_test = std_sc.fit_transform(x_test)

# Building and training the model
model = models.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# https://medium.com/analytics-vidhya/a-complete-guide-to-adam-and-rmsprop-optimizer-75f4502d83be
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

n_epochs = 5
history = model.fit(
    x_train,
    y_train,
    epochs=n_epochs,
    batch_size=16,
    verbose=True,
)

# Evaluating the model
test_mse_score, test_mae_score = model.evaluate(x_test, y_test)
print('MSE: %.2f | MAE: %2.f' % (test_mse_score, test_mae_score))

# Visualising the results
data = [
    {
        'epoch': idx,
        'loss': history.history['loss'][idx],
        'mae': history.history['mae'][idx]
    }
    for idx in range(n_epochs)
]

hip.Experiment.from_iterable(data).to_html('outputs/bh.html')
