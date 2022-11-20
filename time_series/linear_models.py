from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

# Configure Pandas
pd.set_option("display.precision", 2)
pd.set_option("expand_frame_repr", False)
pd.set_option('display.max_rows', None)

# The datasets
ads = pd.read_csv('../assets/ads.csv', index_col=['Time'], parse_dates=['Time'])
currency = pd.read_csv('../assets/currency.csv', index_col=['Time'], parse_dates=['Time'])

data = pd.DataFrame(ads['Ads']).copy()

# Adding the lag at the target variable
data.columns = ['y']

for i in range(6, 25):
    data['lag_{}'.format(i)] = data['y'].shift(periods=i)


def fit_model_with_lag():
    # Composing the dataset for the fitting
    X = data.dropna().drop(['y'], axis=1)
    y = data.dropna()['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    plot_model_results(model, X_train, y_train, y_test, y_pred, plot_intervals=True, plot_anomalies=True)
    plot_model_coefficients(model, X_train)


def fit_advanced_model():
    # Fetching hour and weekday from the dataset
    data.index = pd.to_datetime(data.index)
    data['hour'] = data.index.hour
    data['weekday'] = data.index.weekday
    data['is_weekend'] = data.weekday.isin([5, 6]) * 1

    # Composing the dataset for the fitting
    X = data.dropna().drop(['y'], axis=1)
    y = data.dropna()['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Scaling the data
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    plot_model_results(model, X_train_scaled, y_train, y_test, y_pred, plot_intervals=True, plot_anomalies=True)
    plot_model_coefficients(model, X_train)


def plot_model_results(model, X_train, y_train, y_test, y_pred,  plot_intervals=False, plot_anomalies=False):
    """
    Plots actual and predicted values
    Plots upper and lower bonds
    Plots anomalies
    """

    plt.figure(figsize=(15, 7))

    # Plot actual and predicted values
    plt.plot(y_test.values, label='Actual')
    plt.plot(y_pred, 'g', label='Predictions')

    # Performing CV for calculating Lower and Upper bonds
    tscv = TimeSeriesSplit(n_splits=5)
    cv = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')

    mae = cv.mean() * (-1)
    deviation = cv.std()
    scale = 1.96

    lower = y_pred - (mae + scale * deviation)
    upper = y_pred + (mae + scale * deviation)

    error = mean_absolute_percentage_error(y_pred, y_test).__round__(3)

    # Plot intervals
    if plot_intervals:
        plt.plot(lower, 'r--', label='Lower / Upper Bond', alpha=0.5)
        plt.plot(upper, 'r--', alpha=0.5)

    # Plot anomalies
    if plot_anomalies:
        anomalies = np.array([np.NaN] * len(y_test))

        anomalies[y_test < lower] = y_test[y_test < lower]
        anomalies[y_test > upper] = y_test[y_test > upper]

        plt.plot(anomalies, "ro", markersize=10, label="Anomalies")

    plt.title('Mean absolute percentage error: %s' % error)

    plt.legend()
    plt.grid(True)
    plt.show()


def plot_model_coefficients(model, X_train):
    """
    Plots sorted coefficient values of the model
    """

    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ['coef']
    coefs['abs'] = coefs['coef'].apply(np.abs)
    coefs = coefs.sort_values(by='abs', ascending=False).drop(['abs'], axis=1)

    plt.figure(figsize=(15, 7))

    coefs['coef'].plot(kind='bar')
    plt.title('Model Coefficients')

    plt.grid(True)
    plt.show()


fit_model_with_lag()
fit_advanced_model()
