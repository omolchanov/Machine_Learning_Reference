"""
Neural network intended for multiclass (~30) classification.
Contains re-usable functions for plotting the data, cross-validation and search of model's best parameters
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import sys
import pprint

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

from keras_tuner import RandomSearch, Objective

from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
np.set_printoptions(threshold=np.inf, suppress=True)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
np.set_printoptions(threshold=sys.maxsize, suppress=True)


def _get_dataframe(encoded_values=True, filter_n=0):
    """
    Prepares the dataframe for further use with processing purposes or plotting the data
    :param encoded_values:
        - if True, the encoded and preprocessed features and labels returned
        - if False, the unprocessed data returned
    :param filter_n: filter instances from classes with number of instances larger than or equal to
    :return: processed or unprocessed dataframe
    """

    df = pd.read_csv('../assets/street_alert.csv')
    df = df.groupby('area').filter(lambda x: len(x) >= filter_n)

    if encoded_values is True:
        ENCODED_WEEKDAY = {
            'monday': 1,
            'tuesday': 2,
            'wednesday': 3,
            'thursday': 4,
            'friday': 5,
            'saturday': 6,
            'sunday': 7
        }

        ENCODED_TIME_DAY = {'morning': 0, 'day': 1, 'evening': 2}

        df['encoded_weekday'] = df['weekday'].replace(ENCODED_WEEKDAY)
        df['encoded_time_day'] = df['time_day'].replace(ENCODED_TIME_DAY)

        enc = LabelEncoder()
        enc.fit(df['area'])
        enc_1 = enc.transform(df['area'])

        df = df.drop(['weekday', 'time_day', 'area'], axis=1)

        X = df[['encoded_weekday', 'encoded_time_day']]

        std_sc = StandardScaler()
        X = std_sc.fit_transform(X)

        poly_sc = PolynomialFeatures(degree=3, include_bias=True)
        X = poly_sc.fit_transform(X)

        sampler = SMOTE(k_neighbors=1)
        X, enc_1 = sampler.fit_resample(X, enc_1)

        y = to_categorical(enc_1)

        return X, y

    return df


def _plot_dataframe():
    """
    Plots the scatter plot for observations by weekdays and time of the days for the Street alert dataframe
    """

    df = _get_dataframe(encoded_values=False)

    fig, ax = plt.subplots(2, 1, sharex=True)

    sns.scatterplot(x='area', y='weekday', data=df, ax=ax[0])
    sns.scatterplot(x='area', y='time_day', data=df, c='orange', ax=ax[1])

    fig.suptitle('Samples by weekday and time of the day')
    plt.xticks(rotation=90)
    plt.show()


def get_model() -> Sequential:
    """
    Builds a Keras model basing on the structure of layers, loss function and optimizer
    :return: model
    """

    model = Sequential([
        Dense(512, activation='relu'),
        Dropout(rate=0.25),
        Dense(512, activation='relu'),
        Dropout(rate=0.25),
        Dense(30, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def evaluate_dataframe(df):
    """
    Prints the classes grouped by count
    :param df: - dataframe with non-encoded data
    """

    areas = df.groupby(['area']).size().sort_values(ascending=False).to_dict()

    print('Classes count: ', len(areas.keys()))
    pprint.pprint(areas, sort_dicts=False)


def evaluate_model(model, X, y):
    """
    Fits and evaluates the model with loss and accuracy
    :param model - Keras compiled model
    :param X - matrix with features
    :param y - vector with labels
    """

    history = model.fit(X, y, epochs=800, batch_size=15, verbose=True, validation_split=0.2)
    loss, accuracy = model.evaluate(X, y)

    sns.lineplot(history.history['accuracy']).set(
        title='Results with Artificial Neural Network',
        xlabel='epochs',
        ylabel='accuracy'
    )

    plt.show()

    print('Loss: %.3f | Accuracy: %.3f' % (loss, accuracy))


def find_best_hyperparameters(X, y):
    """
    Computes a set of hyperparameters of NN basing on accuracy metric
    :param X - matrix with features
    :param y - vector with labels
    """

    def build_model(hp):

        model = Sequential([
            Dense(
                units=hp.Int('units_dense_layer_1', min_value=64, max_value=512, step=64),
                activation=hp.Choice('activation_dense_layer_1', ['relu', 'linear'])
            ),
            Dropout(hp.Choice('dropout_rate_1', [0.2, 0.25, 0.3])),
            Dense(
                units=hp.Int('units_dense_layer_2', min_value=64, max_value=512, step=64),
                activation=hp.Choice('activation_dense_layer_2', ['relu', 'linear'])
            ),

            Dense(30, activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    tuner = RandomSearch(
        hypermodel=build_model,
        objective=Objective('accuracy', direction='max'),
        max_trials=200,
        executions_per_trial=2,
        overwrite=True,
        max_consecutive_failed_trials=100,
        directory='saved_models',
        project_name='gs'
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

    tuner.search(X_train, y_train, epochs=2, validation_data=(X_test, y_test), verbose=True)

    print(tuner.results_summary())
    print('\nBest parameters: ', tuner.get_best_hyperparameters()[0].values)


def cross_validate_model(model, X, y):
    """
    Cross-validates Keras model. Plots the lineplot with train and test scores in order to identify if
    the model is overfitted or underfitted

    :param model: - Keras compiled model
    :param X: - matrix with features
    :param y: - vector with labels
    """

    test_scores, train_scores, folds = [], [], []

    k = 12
    cv = KFold(n_splits=k)

    f = 1
    for i_train, i_test in cv.split(X, y):
        print('\nFold #', f)

        X_train = np.take(X, i_train, axis=0)
        X_test = np.take(X, i_test, axis=0)

        y_train = np.take(y, i_train, axis=0)
        y_test = np.take(y, i_test, axis=0)

        model.fit(X_train, y_train, epochs=15, batch_size=15, verbose=True)

        train_score = model.evaluate(X_train, y_train)
        test_score = model.evaluate(X_test, y_test)

        test_scores.append(test_score[1])
        train_scores.append(train_score[1])
        folds.append(f)

        f = f + 1

    print('TEST ==> Mean accuracy: %.3f' % np.mean(test_scores))
    print('TRAIN ==> Mean accuracy: %.3f' % np.mean(train_scores))

    plt.plot(folds, test_scores)
    plt.plot(folds, train_scores)

    plt.title('Cross Validation')
    plt.legend(['Test', 'Train'])

    plt.xlabel('Fold')
    plt.ylabel('Accuracy')

    plt.show()


def plot_confusion_matrix(model, X, y):
    """
    Computes and plots a confusion matrix for Keras NN model
    :param model - Keras compiled model
    :param X - matrix with features
    :param y - vector with UNCODED labels
    """

    assert y.ndim == 1, 'Labels should be uncoded'

    model.fit(X, y, epochs=5, batch_size=15, verbose=True, validation_split=0.2)
    y_pred = model.predict(X)

    cm = confusion_matrix(y, y_pred)

    sns.heatmap(cm, linewidths=.10, annot=True).set(title='Confusion matrix')
    plt.show()


df_o = _get_dataframe(encoded_values=False)
X_o, y_o = _get_dataframe()
model_o = get_model()

# plot_dataframe()

# evaluate_dataframe(df_o)
evaluate_model(model_o, X_o, y_o)
# find_best_hyperparameters(X, y)
# cross_validate_model(model_o, X_o, y_o)
# plot_confusion_matrix(model_o, X_o, y_o)
