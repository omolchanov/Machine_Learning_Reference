"""
Multiouptput regression with mixed types of features and labels
Guideline: https://machinelearningmastery.com/neural-network-models-for-combined-classification-and-regression/
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

from sklearn.metrics import mean_absolute_percentage_error

from keras.models import Sequential
from keras.layers import Dense, Input, Dropout


import pandas as pd
import numpy as np

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
pd.set_option('display.float_format', lambda x: '%.i' % x)


def get_dataframe():
    """
    Reads, composes and encodes the basic dataframe
    :return: dataframe, exercise encoder
    """

    df = pd.read_csv('../assets/fitness_programme.csv', encoding='utf-8', header=None)

    df.columns = [
        'person weight',
        'exercise',
        'exercise weight start',
        'exercise weight end',
        'n_sets',
        'n_repetitons',
        'is_health_problems',
        'bb_type',

        'r_exercise',
        'r_exercise_weight',
        'r_n_sets',
        'r_n_repetitons',
        'r_rest_time'
    ]

    # Encoding string categorial features [exercise, r_exercise, bb_type] to int
    ex_enc = LabelEncoder()

    ex_enc.fit(df['exercise'])
    df['exercise'] = ex_enc.transform(df['exercise'])
    df['r_exercise'] = ex_enc.transform(df['r_exercise'])

    df['bb_type'] = LabelEncoder().fit_transform(df['bb_type'])

    return df, ex_enc


def get_encoded_dataframe_sklearn():
    """
    Encodes continious features ['n_sets', 'n_repetitons', 'r_n_sets', 'r_n_repetitons'] to categorial
    :return: encoded dataframe
    """

    df, _ = get_dataframe()

    continious_features = [
        'n_sets',
        'n_repetitons',

        'r_n_sets',
        'r_n_repetitons'
    ]

    for c in continious_features:
        df[c] = LabelEncoder().fit_transform(df[c])

    return df


def evaluate_sklearn_models():
    """
    Evaluates numerous sklearn models with non-encoded and encoded data.
    Builds the dataframe with results.
    """

    models = (
        KNeighborsClassifier(n_neighbors=2),
        # KNeighborsRegressor(n_neighbors=2),
        DecisionTreeClassifier(),
        DecisionTreeRegressor(),
        MultiOutputClassifier(KNeighborsClassifier()),
        MultiOutputClassifier(DecisionTreeClassifier()),
        # MultiOutputRegressor(KNeighborsRegressor()),
        MultiOutputRegressor(DecisionTreeRegressor())
    )

    def evaluate(df):

        X = df.iloc[:, :8]
        y = df.iloc[:, 8:]

        mapes = []

        for m in models:
            m.fit(X, y)

            y_pred = m.predict(X)

            mape = mean_absolute_percentage_error(y, y_pred) * 100
            mapes.append(mape)

        return mapes

    non_encoded_df, _ = get_dataframe()
    encoded_df = get_encoded_dataframe_sklearn()

    results = pd.DataFrame({
        'MAPE (non-encoded continious features)': evaluate(non_encoded_df),
        'MAPE (encoded continious features)': evaluate(encoded_df)
    }, index=models)

    print('\n', results)


def get_nn_dataframe():
    """
    Encodes continious features ['n_sets', 'n_repetitons', 'r_n_sets', 'r_n_repetitons'] to categorial
    :return: encoded dataframe
    """

    df, _ = get_dataframe()

    continious_features = [
        'n_sets',
        'n_repetitons',

        'r_n_sets',
        'r_n_repetitons'
    ]

    for c in continious_features:
        df[c] = LabelEncoder().fit_transform(df[c])

    return df


def evaluate_nn_model():
    """
    Builds and evaluates a Keras NN model
    """

    def evaluate(df):

        X = df.iloc[:, :8]
        Y = df.iloc[:, 8:]

        model = Sequential([
            Input(shape=(8,)),
            Dense(512, activation='relu'),
            Dropout(0.5),

            Dense(256, activation='relu'),
            Dropout(0.5),

            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(5, activation='softmax'),
        ])

        model.compile(loss='mape', optimizer='adam')

        results = {}

        for i in range(3):
            history = model.fit(X, Y, epochs=150, batch_size=5, verbose=False)

            loss = np.mean(history.history['loss'])
            results['Iteration ' + str(i)] = loss

        return results

    non_encoded_df, _ = get_dataframe()
    encoded_df = get_nn_dataframe()

    results = pd.DataFrame({
        'MAPE (non-encoded continious features)': evaluate(non_encoded_df),
        'MAPE (encoded continious features)': evaluate(encoded_df)
    })

    print(results)


def compose_programme_sklearn():
    df, ex_enc = get_dataframe()

    X = df.iloc[:, :8]
    y = df.iloc[:, 8:]

    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(X, y)

    X_pred = [
        [100, 4, 70, 80, 4, 8, 0, 1],
        [100, 3, 80, 90, 3, 10, 0, 1],
        [100, 7, 120, 125, 3, 10, 0, 1]
    ]

    y_pred = model.predict(X_pred)

    programme = pd.DataFrame(columns=[
        'r_exercise',
        'r_exercise_weight',
        'r_n_sets',
        'r_n_repetitons',
        'r_rest_time'
    ])

    for ex in y_pred:
        dec_ex = []

        for i, v in enumerate(ex):

            # Decoding 'r_exercise' feature
            if i == 0:
                dec_ex.append(ex_enc.inverse_transform([v])[0])
                continue

            dec_ex.append(v)

        programme.loc[len(programme)] = dec_ex

    print(programme)


evaluate_sklearn_models()
compose_programme_sklearn()

evaluate_nn_model()
