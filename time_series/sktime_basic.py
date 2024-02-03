import warnings

import numpy as np
import pandas as pd

from sktime.datasets import load_airline
from sktime.datasets import load_longley

from sktime.utils.plotting import plot_series

from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.var import VAR
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.theta import ThetaForecaster

from matplotlib import pyplot as plt

# Configuring pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)

warnings.filterwarnings("ignore")


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


def multivariative_forecast(fh):
    """
    Multivariative forecaster implemented with VAR (vector auto-regression) algorithm - multivariative
    (depended on values of other variables)
    And ARIMA forecaster - univariative (not depended on other variables.)
    """

    _, y = load_longley()
    y = y[['POP', 'UNEMP']]

    fc = VAR()
    fc.fit(y, fh=range(1, fh + 1))

    y_pred = fc.predict()
    print('VAR model')
    print(y_pred)

    fc = ARIMA()
    fc.fit(y, fh=range(1, fh + 1))

    y_pred = fc.predict()
    print('ARIMA model')
    print(y_pred)


def propabilistic_forecast(fh):
    """
    predict_interval produces symmetric prediction intervals, for example, a coverage of 0.9 returns a
    “lower” forecast at quantile 0.5 - coverage/2 = 0.05, and an
    “upper” forecast at quantile 0.5 + coverage/2 = 0.95.

    The prediction variance is the variance of observations. It is a combination of the variance of the
    response (standard deviation squared) and the variance of the predicted mean
    (standard error squared).

    Forecast distributions describes the probability of observing possible future values using the
    fitted model. The point forecast is the mean of this distribution.
    """

    y = load_airline()

    fh = np.arange(1, fh + 1)

    fc = ThetaForecaster(sp=12)
    fc.fit(y, fh=fh)

    y_pred = fc.predict()
    y_pred_ints = fc.predict_interval(coverage=0.5)
    print('INTERVALS \n', y_pred_ints)

    y_pred_quant = fc.predict_quantiles(alpha=[0.25, 0.5, 0.75])
    print('QUANTILES \n', y_pred_quant)

    y_pred_var = fc.predict_var()
    print('VARIANCE \n', y_pred_var)

    y_pred_proba = fc.predict_proba()
    print('PROBA \n', y_pred_proba)

    plot_series(
        y,
        y_pred,
        labels=["y", "y_pred"],
        pred_interval=y_pred_ints,
        title='Prediction intervals'
    )
    plt.show()


# simple_pseudo_classification()
# airline_naive_forecast()
# multivariative_forecast(5)
propabilistic_forecast(6)
