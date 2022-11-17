from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Configure Pandas
pd.set_option("display.precision", 2)
pd.set_option("expand_frame_repr", False)
pd.set_option('display.max_rows', None)


# The datasets
ads = pd.read_csv('../assets/ads.csv', index_col=['Time'], parse_dates=['Time'])
currency = pd.read_csv('../assets/currency.csv', index_col=['Time'], parse_dates=['Time'])


class HoltWinters:
    """
    Holt-Winters model with the anomalies detection using Brutlag method
     - series - initial time series
     - slen - length of a season
     - alpha, beta, gamma - Holt-Winters model coefficients
     - n_preds - predictions horizon
     - scaling_factor - sets the width of the confidence interval by Brutlag (usually takes values from 2 to 3)
    """

    def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96, verbose=False):
        self.series = series
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor
        self.verbose = verbose

        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []

    def initial_trend(self):
        sum = 0.0

        for i in range(self.slen):
            sum += float(self.series[i + self.slen] - self.series[i]) / self.slen

        return sum / self.slen

    def initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series) / self.slen)

        # let's calculate season averages
        for j in range(n_seasons):
            season_averages.append(
                sum(self.series[self.slen * j: self.slen * j + self.slen])
                / float(self.slen)
            )

        # let's calculate initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += (
                        self.series[self.slen * j + i] - season_averages[j]
                )
            seasonals[i] = sum_of_vals_over_avg / n_seasons

        return seasonals

    def triple_exponential_smoothing(self):

        seasonals = self.initial_seasonal_components()

        for i in range(len(self.series) + self.n_preds):
            if i == 0:  # components initialization
                smooth = self.series[0]
                trend = self.initial_trend()

                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i % self.slen])

                self.PredictedDeviation.append(0)

                self.UpperBond.append(
                    self.result[0] + self.scaling_factor * self.PredictedDeviation[0]
                )

                self.LowerBond.append(
                    self.result[0] - self.scaling_factor * self.PredictedDeviation[0]
                )

                continue

            if i >= len(self.series):  # predicting
                m = i - len(self.series) + 1
                self.result.append((smooth + m * trend) + seasonals[i % self.slen])

                # when predicting we increase uncertainty on each step
                self.PredictedDeviation.append(self.PredictedDeviation[-1] * 1.01)

            else:
                val = self.series[i]
                last_smooth, smooth = (
                    smooth,
                    self.alpha * (val - seasonals[i % self.slen])
                    + (1 - self.alpha) * (smooth + trend),
                )
                trend = self.beta * (smooth - last_smooth) + (1 - self.beta) * trend
                seasonals[i % self.slen] = (
                        self.gamma * (val - smooth)
                        + (1 - self.gamma) * seasonals[i % self.slen]
                )
                self.result.append(smooth + trend + seasonals[i % self.slen])

                # Deviation is calculated according to Brutlag algorithm.
                self.PredictedDeviation.append(
                    self.gamma * np.abs(self.series[i] - self.result[i])
                    + (1 - self.gamma) * self.PredictedDeviation[-1]
                )

            self.UpperBond.append(
                self.result[-1] + self.scaling_factor * self.PredictedDeviation[-1]
            )

            self.LowerBond.append(
                self.result[-1] - self.scaling_factor * self.PredictedDeviation[-1]
            )

            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i % self.slen])

        # Printing Results
        if self.verbose:
            print('result', self.result, '\n')
            print('Smooth', self.Smooth, '\n')
            print('Season', self.Season, '\n')
            print('Trend', self.Trend, '\n')
            print('Predicted Deviation', self.PredictedDeviation, '\n')
            print('Upper Bond', self.UpperBond, '\n')
            print('Lower Bond', self.LowerBond, '\n')


def time_series_cv_score(params, series, slen=24, n_splits=3):
    """
    Returns error on cross validation. Loss function is MAE

    - params - vector of parameters for optimization
    - series - dataset with timeseries
    - slen - season length for Holt-Winters model
    """

    errors = []

    values = series.values
    alpha, beta, gamma = params

    cv = TimeSeriesSplit(n_splits)

    # iterating over folds, train model on each, forecast and calculate error
    for train, test in cv.split(values):
        model = HoltWinters(
            series=values[train],
            slen=slen,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            n_preds=len(test),
            verbose=False
        )

        model.triple_exponential_smoothing()

        predictions = model.result[-len(test):]
        actual = values[test]

        error = mean_absolute_error(actual, predictions)
        errors.append(error)

        return np.mean(np.array(errors))


def predict_with_optimal_params(series, slen):
    """
    Finds optimal alpha, beta, gamma parameters for Holt Winters model
    Makes a prediction with the optimal parameters
    """

    X = series[:-24]

    # Initialising basic parameters
    x = np.array([0.13, 0.047, 0.0])

    # Minimize the loss function. Finding the optimal parameters
    # opt = minimize(
    #     time_series_cv_score,
    #     x0=x,
    #     args=(X, mean_squared_log_error),
    #     method='TNC',
    #     bounds=((0, 1), (0, 1), (0, 1))
    # )

    alpha_final, beta_final, gamma_final = x

    # Training and predicting
    model = HoltWinters(
        X,
        slen=slen,
        alpha=alpha_final,
        beta=beta_final,
        gamma=gamma_final,
        n_preds=50,
        scaling_factor=3,
    )

    model.triple_exponential_smoothing()
    return model


def plot_holt_winters(series, slen, plot_intervals=False, plot_anomalies=False, plot_deviations=False):
    """
    Plots results of Holt Winters exponential smoothing

    """

    # Model Object
    model = predict_with_optimal_params(series, slen)

    actual = series.values
    predictions = model.result
    error = mean_absolute_percentage_error(actual, predictions[:len(series)]).__round__(3)

    # Plot actual values and predictions
    plt.figure(figsize=(15, 8))

    plt.plot(actual, 'b', label='Actual values')
    plt.plot(predictions, 'g', label='Predicted values')
    plt.title('SLen: %s. MA Percentage Error: %s' % (slen, error))

    plt.vlines(
        len(series),
        ymin=min(model.LowerBond),
        ymax=max(model.UpperBond),
        linestyles="dashed",
    )

    # Plot intervals
    if plot_intervals:
        plt.plot(model.UpperBond, 'r--', alpha=0.5, label='Upper/Lower Bond')
        plt.plot(model.LowerBond, 'r--', alpha=0.5)

        plt.fill_between(
            x=range(0, len(model.result)),
            y1=model.UpperBond,
            y2=model.LowerBond,
            alpha=0.2,
            color="grey",
        )

    # Plot anomalies
    if plot_anomalies:
        anomalies = np.array([np.NaN] * len(series))

        anomalies[series.values < model.LowerBond[: len(series)]] = \
            series.values[series.values < model.LowerBond[: len(series)]]

        anomalies[series.values > model.UpperBond[: len(series)]] = \
            series.values[series.values > model.UpperBond[: len(series)]]

        plt.plot(anomalies, 'ro', markersize=10, label='Anomalies')

    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    # Plot deviations
    if plot_deviations:
        plt.figure(figsize=(15, 8))

        plt.plot(model.PredictedDeviation, 'b', label='Predicted Deviation')
        plt.title("Brutlag's predicted deviation")

        plt.legend()
        plt.grid(True)
        plt.show()


plot_holt_winters(
    currency['GEMS_GEMS_SPENT'],
    30,
    plot_intervals=False,
    plot_anomalies=True,
    plot_deviations=True
)
