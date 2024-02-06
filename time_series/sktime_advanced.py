import pprint
import warnings

import pandas as pd

from sktime.datasets import load_airline

from sktime.forecasting.base import ForecastingHorizon

from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError, MeanSquaredError
from sktime.split import temporal_train_test_split, SlidingWindowSplitter, ExpandingWindowSplitter
from sktime.forecasting.compose import make_reduction, RecursiveTabularRegressionForecaster
from sktime.forecasting.model_selection import ForecastingGridSearchCV

from sklearn.neighbors import KNeighborsRegressor
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import ARIMA, AutoARIMA

from sktime.utils.plotting import plot_series
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')

# Configuring pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)


def reduction():
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=36)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    regressor = KNeighborsRegressor(n_neighbors=1)
    fc = make_reduction(regressor, window_length=15, strategy='recursive')

    fc.fit(y_train)
    y_pred = fc.predict(fh)

    pprint.pprint(fc.get_params())
    print('MAPE: %.2f' % MeanAbsolutePercentageError().evaluate(y_test, y_pred, symmetric=False))

    plot_series(y_train, y_test, y_pred, labels=['y_train', 'y_test', 'y_pred'])
    plt.show()


def get_best_params(fc):
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=12)

    fh = range(1, 12 + 1)
    param_grid = {}

    if isinstance(fc, NaiveForecaster):
        param_grid = {
            'strategy': ['last', 'mean', 'drift'],
            'window_length': [7, 12, 15]
        }

    if isinstance(fc, RecursiveTabularRegressionForecaster):
        param_grid = {
            'window_length': [7, 12, 15],
            'estimator__n_neighbors': range(1, 5 + 1)
        }

    # cv = SlidingWindowSplitter(initial_window=72, window_length=20)
    cv = ExpandingWindowSplitter(step_length=12, fh=fh, initial_window=72)

    gscv = ForecastingGridSearchCV(
        fc,
        strategy='refit',
        cv=cv,
        param_grid=param_grid,
        scoring=MeanAbsolutePercentageError()
    )

    gscv.fit(y_train)
    y_pred = gscv.predict(fh)

    mape = MeanAbsolutePercentageError().evaluate(y_test, y_pred, symmetric=False)

    print(gscv.cv_results_)
    print('Best params: {}, \nBest MAPE: {}'.format(gscv.best_params_, gscv.best_score_.round(3)))
    print('MAPE: %.3f' % mape)

    plot_series(
        y_train, y_test, y_pred,
        labels=['y_train', 'y_test', 'y_pred'],
    )
    plt.show()


fc = make_reduction(KNeighborsRegressor(), strategy='recursive')
get_best_params(fc)
