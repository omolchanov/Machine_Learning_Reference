# Guidelines:
# https://towardsdatascience.com/visualizing-backpropagation-in-neural-network-training-2647f5977fdb
# https://facebookresearch.github.io/hiplot/
# https://hiranh.medium.com/visualize-keras-neural-networks-with-netron-9d3f9b3e4b5a

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import itertools

from keras import layers
from keras import models
from keras.datasets import boston_housing
from keras.callbacks import LambdaCallback

from tensorflow.keras.utils import plot_model

from sklearn.preprocessing import StandardScaler

import hiplot as hip
import numpy as np

import netron

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

weights = []
print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: weights.append(model.get_weights()))


# https://medium.com/analytics-vidhya/a-complete-guide-to-adam-and-rmsprop-optimizer-75f4502d83be
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

n_epochs = 5
history = model.fit(
    x_train,
    y_train,
    epochs=n_epochs,
    batch_size=16,
    verbose=True,
    callbacks=[print_weights]
)


# Evaluating the model
test_mse_score, test_mae_score = model.evaluate(x_test, y_test)
print('MSE: %.2f | MAE: %2.f' % (test_mse_score, test_mae_score))


# Visualising the results
# HiPlot
# To start the HiPlot server run "python -m hiplot"
data = [
    {
        'epoch': i,
        'loss': history.history['loss'][i],
        'mae': history.history['mae'][i],
        # 'weights': [round(x, 2) for x in itertools.chain.from_iterable(weights[i][0].tolist())],
        # 'bias': [round(x, 2) for x in (weights[i][1].tolist())]
    }
    for i in range(n_epochs)
]

hip.Experiment.from_iterable(data).to_html('outputs/bp_model.html')

# PLotting the model's architecture
plot_model(model, to_file='outputs/keras_model_plot.png', show_shapes=True, show_layer_names=True)

# Saving the model
# Online visualizer: https://netron.app/
MODEL_PATH = 'saved_models/bp_model.keras'

model.save(MODEL_PATH)
netron.start(MODEL_PATH)
