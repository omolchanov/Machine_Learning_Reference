from linear_models import plot_model_results, plot_model_coefficients

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV

from xgboost import XGBRegressor

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

# Configure Pandas
pd.set_option("display.precision", 2)
pd.set_option("expand_frame_repr", False)
pd.set_option('display.max_rows', None)

# The datasets
ads = pd.read_csv('../assets/ads.csv', index_col=['Time'], parse_dates=['Time'])

# Adding hour, weekday and weekend features. Adding lag
data = pd.DataFrame(ads['Ads']).copy()
data.columns = ['y']

for i in range(6, 25):
    data['lag_{}'.format(i)] = data['y'].shift(periods=i)

data.index = pd.to_datetime(data.index)
data['hour'] = data.index.hour
data['weekday'] = data.index.weekday
data['is_weekend'] = data.weekday.isin([5, 6]) * 1

X = data.dropna().drop(['y'], axis=1)
y = data.dropna()['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Scaling the dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def plot_correlations():
    """
    Plotting correlation between features
    """

    plt.figure(figsize=(15, 7))
    sns.heatmap(X_train.corr())
    plt.show()


def regularize_Ridge():
    model = RidgeCV(cv=5)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    plot_model_results(model, X_train_scaled, y_train, y_test, y_pred, plot_intervals=True, plot_anomalies=True)
    plot_model_coefficients(model, X_train)


def regularize_Lasso():
    model = LassoCV(cv=5)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    plot_model_results(model, X_train_scaled, y_train, y_test, y_pred, plot_intervals=True, plot_anomalies=True)
    plot_model_coefficients(model, X_train)


def regularize_boost():
    model = XGBRegressor()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    plot_model_results(model, X_train_scaled, y_train, y_test, y_pred, plot_intervals=True, plot_anomalies=True)


plot_correlations()
regularize_Ridge()
regularize_Lasso()
regularize_boost()
