import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

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


plot_moving_average(ads, plot_bonds=True, plot_anomalies=True)
calculate_moving_average(ads)
plotting_data()
