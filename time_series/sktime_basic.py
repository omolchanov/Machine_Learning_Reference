import numpy as np
import pandas as pd

from sktime.datasets import load_airline
from sktime.utils.plotting import plot_series
from sktime.forecasting.naive import NaiveForecaster

from matplotlib import pyplot as plt

# Configuring pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)


def simple_pseudo_classification():
    """
    Simple classifier composed with exogeneous data
    """

    days = []

    for j in range(1, 5):
        for i in range(1, 8):
            days.append(i)

    areas = range(1, 29)

    df = pd.DataFrame(areas, index=days, columns=['area'])
    X = pd.DataFrame(df.index)
    y = df.values

    fh = np.arange(1, 8)

    fc = NaiveForecaster(strategy='last', sp=7)
    fc.fit(y, X=X, fh=fh)
    y_pred = fc.predict(X=X, fh=fh)
    plt.scatter(X.index, y)
    plt.scatter(fh, y_pred)

    plt.show()


def airline_naive_forecast():
    """
    Naive forecaster applied to the airline dataset.
    Relative forecast horizon was used
    """

    y = load_airline()

    # Specifying forecasting horizon
    fh = np.arange(1, 12 + 1)

    fc = NaiveForecaster(strategy='last', sp=12)
    fc.fit(y, fh=fh)

    y_pred = fc.predict(fh)

    plot_series(y, y_pred)
    plt.show()


# simple_pseudo_classification()
airline_naive_forecast()
