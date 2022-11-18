import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_percentage_error
from itertools import product
from tqdm import tqdm

import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import warnings
warnings.filterwarnings('ignore')

# Configure Pandas
pd.set_option("display.precision", 2)
pd.set_option("expand_frame_repr", False)
pd.set_option('display.max_rows', None)

# The datasets
ads = pd.read_csv('../assets/ads.csv', index_col=['Time'], parse_dates=['Time'])
currency = pd.read_csv('../assets/currency.csv', index_col=['Time'], parse_dates=['Time'])


def plot_stationarity_process(n_samples, rho=0):
    x = w = np.random.normal(size=n_samples)

    for t in range(n_samples):
        x[t] = rho * x[t - 1] + w[t]

    with plt.style.context("bmh"):
        plt.figure(figsize=(15, 7))

        plt.plot(x)
        plt.title('Rho %s \n Dickey-Fuller p-value: %s' % (rho, round(sm.tsa.stattools.adfuller(x)[1], 3)))

        plt.show()


plot_stationarity_process(1000, 1)


# Plotting series, AFC, PACF
# Manual: https://towardsdatascience.com/interpreting-acf-and-pacf-plots-for-time-series-forecasting-af0d6db4061c
def plot_time_series(series, lags):
    p_value = round(sm.tsa.stattools.adfuller(series)[1], 5)

    plt.figure(figsize=(15, 7))

    plt.plot(series, 'b', label='Actual value')
    plt.title('Dickey-Fuller p-value: %s' % p_value)

    plot_acf(series, lags=lags)
    plot_pacf(series, lags=lags, method='ols')

    plt.show()


# Initial plotting
plot_time_series(ads, 60)
ads_diff = ads['Ads'] - ads['Ads'].shift(24)

plot_time_series(ads_diff[24:], lags=60)

ads_diff = ads_diff - ads_diff.shift(1)
plot_time_series(ads_diff[24 + 1 :], lags=60)

plot_time_series(ads_diff[24 + 1:], lags=60)


# Article: https://coolstatsblog.com/2013/08/14/using-aic-to-test-arima-models-2/
def optimize_SARIMA(parameters, d, D, s):
    """
    Return dataframe with parameters and corresponding AIC

    parameters_list - list with (p, q, P, Q) tuples
    d - integration order in ARIMA model
    D - seasonal integration order
    s - length of season
    """

    results = []
    best_aic = float('inf')

    for p in tqdm(parameters):
        try:
            model = sm.tsa.statespace.SARIMAX(
                ads.Ads,
                order=(p[0], d, p[1]),
                seasonal_order=(p[2], D, p[3], s),
            ).fit(disp=-1)

            results.append([p, model.aic])

        except:
            continue

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']

    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by="aic", ascending=True).reset_index(drop=True)

    return result_table


def plot_SARIMA(series, model, n_steps):
    """
    Plots model vs predicted values
        series - dataset with timeseries
        model - fitted SARIMA model
        n_steps - number of steps to predict in the future
    """

    # adding model values
    data = series.copy()
    data.columns = ['actual']
    data['arima_model'] = model.fittedvalues
    # making a shift on s+d steps, because these values were unobserved by the model
    # due to the differentiating
    data['arima_model'][: s + d] = np.NaN

    # forecasting on n_steps forward
    forecast = model.predict(start=data.shape[0], end=data.shape[0] + n_steps)
    forecast = data['arima_model'].append(forecast)

    # calculate error, again having shifted on s+d steps from the beginning
    error = mean_absolute_percentage_error(
        data["actual"][s + d:], data["arima_model"][s + d:]
    )

    # Plotting
    plt.figure(figsize=(15, 7))

    plt.plot(forecast, color='r', label='model')
    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.plot(data.actual, label='actual')

    plt.title('Mean Absolute Percentage Error: {0:.2f}%'.format(error))

    plt.legend()
    plt.grid(True)
    plt.show()


# Setting initial values and some bounds for them
ps = range(2, 5)
d = 1
qs = range(2, 5)
Ps = range(0, 2)
D = 1
Qs = range(0, 2)
s = 24

# creating list with all the possible combinations of parameters
parameters = list(product(ps, qs, Ps, Qs))

# Finding optimal parameters for SARIMA
result_table = optimize_SARIMA(parameters, d, D, s)
print(result_table.head(5))

# Set the parameters that give the lowest AIC
p, q, P, Q = result_table.parameters[0]
best_model = sm.tsa.statespace.SARIMAX(ads.Ads, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=-1)
print(best_model.summary())

# Plotting ACF, PASF
plot_time_series(best_model.resid[24 + 1:], lags=60)

# Plotting predictions with SARIMA model
plot_SARIMA(ads, best_model, 50)
