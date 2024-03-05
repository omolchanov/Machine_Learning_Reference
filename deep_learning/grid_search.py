# Guidelines:
# https://keras.io/guides/keras_tuner/getting_started/
# https://medium.com/@viniciusqroz/using-keras-tuner-to-find-the-best-parameters-for-your-neural-network-model-2dc02e0a1203

"""
Grid search of the best model's hyperparameters with Keras tuner
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from keras.models import Sequential
from keras.layers import Dense

from keras_tuner import RandomSearch, Objective

from sklearn.model_selection import train_test_split

import pandas as pd

df = pd.read_csv('../assets/boston_houses.csv', header=None, sep='\s+')

X = df.iloc[:, 0:13]
y = df.iloc[:, 13]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)


def build_model(hp) -> Sequential:
    """
    Builds a model with a particular set of hyperparameters

    :param hp: set of hyperparameters
    :return: Keras model with particular set of hyperparameters
    """

    model = Sequential([
        Dense(
            units=hp.Int('units_dense_layer_1', min_value=1, max_value=3, step=1),
            activation=hp.Choice('activation_dense_layer_1', ['relu', 'linear'])
        ),

        Dense(
            units=hp.Int('units_dense_layer_2', min_value=1, max_value=3, step=1),
            activation=hp.Choice('activation_dense_layer_2', ['sigmoid', 'tanh'])
        )
    ])

    model.compile(
        loss='mean_absolute_error',
        optimizer=hp.Choice('optimizer', values=['adam', 'SGD']),
        metrics=['r2_score'])

    return model


tuner = RandomSearch(
    hypermodel=build_model,
    objective=Objective('val_loss', direction='min'),
    max_trials=100,
    executions_per_trial=1,
    overwrite=True,
    directory='saved_models',
    project_name='gs',
)


# Printing the searchable parameters and the results
print(tuner.search_space_summary())

tuner.search(X_train, y_train, epochs=1, validation_data=(X_test, y_test), verbose=True)

print(tuner.results_summary())
print('\nBest parameters: ', tuner.get_best_hyperparameters()[0].values)

# Building a model with the best parameters
best_model = build_model(tuner.get_best_hyperparameters()[0])
best_model.fit(X, y, epochs=20)
loss, r2 = best_model.evaluate(X, y)
print(loss, r2)
