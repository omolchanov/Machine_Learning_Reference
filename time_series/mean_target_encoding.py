from linear_models import plot_model_results, plot_model_coefficients

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import pandas as pd

# Configure Pandas
pd.set_option("display.precision", 2)
pd.set_option("expand_frame_repr", False)
pd.set_option('display.max_rows', None)

# The datasets
ads = pd.read_csv('../assets/ads.csv', index_col=['Time'], parse_dates=['Time'])


def encode_mean_value(data, cat_feature, values):
    """
    Encodes categorical features with mean values
    """

    return dict(data.groupby(cat_feature)[values].mean())


def plot_mean_values(data):
    averages = encode_mean_value(data, 'hour', 'y')

    plt.figure(figsize=(15, 7))

    pd.DataFrame.from_dict(averages, orient="index")[0].plot()
    plt.title('Hours Averages')

    plt.grid('True')
    plt.show()


if __name__ == '__main__':

    # Adding hour, weekday and weekend features. Adding lag
    data = pd.DataFrame(ads['Ads']).copy()
    data.columns = ['y']

    for i in range(6, 25):
        data['lag_{}'.format(i)] = data['y'].shift(periods=i)

    data.index = pd.to_datetime(data.index)
    data['hour'] = data.index.hour
    data['weekday'] = data.index.weekday
    data['is_weekend'] = data.weekday.isin([5, 6]) * 1

    # Plotting mean values
    plot_mean_values(data)

    # Calculating averages on train set only
    test_size = 0.3
    test_index = int(len(data.dropna()) * (1 - test_size))

    data['weekday_average'] = list(map(encode_mean_value(data[:test_index], 'weekday', 'y').get, data['weekday']))
    data['hour_average'] = list(map(encode_mean_value(data[:test_index], 'hour', 'y').get, data['hour']))

    # Dropping encoded variables
    data.drop(['hour', 'weekday'], axis=1, inplace=True)

    # Building model and plot the results
    X = data.dropna().drop(['y'], axis=1)
    y = data.dropna()['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Scaling data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    plot_model_results(model, X_train_scaled, y_train, y_test, y_pred, plot_intervals=True, plot_anomalies=True)
    plot_model_coefficients(model, X_train)
