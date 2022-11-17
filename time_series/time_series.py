from sklearn.metrics import mean_absolute_error

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Configure Pandas
pd.set_option("display.precision", 2)
pd.set_option("expand_frame_repr", False)
pd.set_option('display.max_rows', None)

ads = pd.read_csv('assets/ads.csv', index_col=['Time'], parse_dates=['Time'])
currency = pd.read_csv('assets/currency.csv', index_col=['Time'], parse_dates=['Time'])


def plotting_data():
    # Plotting Ads data
    plt.figure(figsize=(12, 6))

    plt.plot(ads['Ads'])
    plt.title('Ads watched (hourly data)')

    plt.grid(True)
    plt.show()

    # Plotting Gems spent data
    plt.figure(figsize=(15, 6))

    plt.plot(currency['GEMS_GEMS_SPENT'])
    plt.title('In-game currency spent (daily data)')
    plt.grid(True)
    plt.show()


def calculate_moving_average(x, n=24):
    """
    Calculating moving average for the subset of dataset
    Manual: https://www.statisticshowto.com/probability-and-statistics/statistics-definitions/moving-average/
    """
    print(np.average(x[-n:]))


def calculate_weighted_average(x, weights, n=24):
    assert (np.sum(weights).round(1) == 1), 'Weights sum should equal to 1. Now is %s' % (np.sum(weights))

    result = 0

    for i in range(len(weights)):
        result += np.average(x[-n:]) * weights[i]

    print(result.__round__(0))


def plot_moving_average(series, window=24, plot_bonds=False, plot_anomalies=False):
    scale = 1.96

    rolling_mean = series.rolling(window).mean()

    plt.figure(figsize=(17, 9))

    # Plotting actual value
    plt.plot(series[window:], 'b', label='Actual values')

    # Plotting moving mean
    plt.plot(rolling_mean, 'g', label='Rolling mean')
    plt.title('Moving average. Window size = %s' % window)

    # Calculating lower and upper bonds
    mae = mean_absolute_error(series[window:], rolling_mean[window:])
    std = np.std(series[window:] - rolling_mean[window:])

    lower_bond = rolling_mean - (mae + scale * std)
    upper_bond = rolling_mean + (mae + scale * std)

    # Plotting lower and upper bonds
    if plot_bonds:
        plt.plot(lower_bond, 'r--', label='Lower Bond')
        plt.plot(upper_bond, 'r--', label='Upper Bond')

    # Plotting anomalies
    if plot_anomalies:

        # Simulating anomalies
        ads.iloc[-5] = ads.iloc[-5] * 0.2
        ads.iloc[25] = ads.iloc[25] * 2.75

        anomalies = pd.DataFrame(index=series.index, columns=series.columns)
        anomalies[series < lower_bond] = series[series < lower_bond]
        anomalies[series > upper_bond] = series[series > upper_bond]

        plt.plot(anomalies, 'ro', markersize=10)

    plt.grid(True)
    plt.legend()
    plt.show()


def plot_exponential_smoothing(x, a):
    """
    a - smoothing parameter
    """

    result = [x.iloc[0]]

    for n in range(1, len(x)):
        result.append(a * x.iloc[n] + (1 - a) * result[n - 1])

    plt.figure(figsize=(15, 7))

    plt.plot(x.values, 'b', label='Actual values')
    plt.plot(result, 'g', label='Alpha {}'.format(a))

    plt.title('Exponential Smoothing')
    plt.legend()

    plt.axis()
    plt.grid(True)
    plt.show()


def plot_double_exponential_smoothing(x, a, b):
    """
    a - smoothing parameter for level
    b - smoothing parameter for trend
    """

    result = [x.iloc[0]]

    for n in range(1, len(x) + 1):
        if n == 1:
            level, trend = x.iloc[0], x.iloc[1] - x.iloc[0]

        # Forecasting
        if n >= len(x):
            value = result[-1]

        else:
            value = x.iloc[n]

        last_level, level = level, a * value + (1 - a) * (level + trend)
        trend = b * (level - last_level) + (1 - b) * trend

        result.append(level + trend)

    plt.figure(figsize=(15, 7))

    plt.plot(x.values, 'b', label='Actual values')
    plt.plot(result, 'g', label='Alpha: {} Beta: {}'.format(a, b))

    plt.title('Double Exponential Smoothing')
    plt.legend()

    plt.axis()
    plt.grid(True)
    plt.show()


plotting_data()

calculate_weighted_average(ads, [0.6, 0.3, 0.1])
calculate_moving_average(ads)

plot_moving_average(currency, plot_bonds=True, plot_anomalies=True)
plot_exponential_smoothing(ads, 0.05)
plot_double_exponential_smoothing(currency, 0.02, 0.9)
