import pprint
import pandas as pd

from sktime.registry import all_estimators, all_tags

from sktime.forecasting.base import ForecastingHorizon

from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.bats import BATS
from sktime.forecasting.tbats import TBATS
from sktime.forecasting.structural import UnobservedComponents

from sktime.datasets import load_airline

from sktime.split import temporal_train_test_split, ExpandingWindowSplitter
from sktime.forecasting.model_evaluation import evaluate

from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

from sktime.utils.plotting import plot_series
import seaborn as sns
from matplotlib import pyplot as plt

# Configuring pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)

df = all_estimators('forecaster', as_dataframe=True)


def print_all_estimators():
    print(df['name'])


def get_forecaster_tags():
    fcs = [NaiveForecaster(), AutoARIMA(), ARIMA()]

    for fc in fcs:
        print(fc)
        pprint.pprint(fc.get_tags())


def get_forecasters_with_tags(tags):
    df = all_estimators('forecaster', as_dataframe=True, filter_tags=tags)
    print(df)


def get_all_forecasters():
    df = all_tags(estimator_types='forecaster', as_dataframe=True)
    print(df)


def evaluate_forecaster(forecasters):
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=12)

    fh = ForecastingHorizon(y_test.index, is_relative=False)

    for fc in forecasters:
        fc.fit(y_train)
        y_pred = fc.predict(fh)

        # Cross validation
        cv = ExpandingWindowSplitter(step_length=12, fh=range(1, 12 + 1), initial_window=72)
        cv_results = evaluate(forecaster=fc, y=y, cv=cv, strategy='refit', return_data=True)

        # MAPE and fit time calculation
        mape = mean_absolute_percentage_error(y_test, y_pred, symmetric=False)
        fit_time = cv_results['fit_time'].sum()

        # Print results
        print(fc)
        print('MAPE: %.3f' % mape)
        print('Mean Fit time: %.4f' % fit_time)
        print(cv_results.iloc[:, :5])

        # Print fitted parameters
        pprint.pprint(fc.get_fitted_params())

        # Plotting
        fig, ax = plt.subplots(2,  figsize=(15, 12))

        # CV results
        sns.barplot(
            x=range(0, cv_results.shape[0]),
            y=cv_results['test_MeanAbsolutePercentageError'],
            ax=ax[0]
        ).set(
            title=[fc, mape.__round__(3), fit_time.__round__(4)],
            xlabel='Fold',
            ylabel='MAPE'
        )

        # Actual and predicted data
        plot_series(
            y_train,
            y_test,
            y_pred,
            ax=ax[1],
            labels=['y_train', 'y_test', 'y_pred']
        )

        plt.show()

# print_all_estimators()
# get_forecaster_tags()
# get_forecasters_with_tags({'requires-fh-in-fit': True})
# get_all_forecasters()


fcs = [
    BATS(sp=12, use_trend=True, use_box_cox=False),
    TBATS(sp=12, use_trend=True, use_box_cox=False),
    AutoARIMA(sp=12, suppress_warnings=True),
    ExponentialSmoothing(trend='add', seasonal='additive', sp=12),
    AutoETS(auto=True, sp=12, n_jobs=-1),
    ARIMA(order=(1, 1, 0), seasonal_order=(0, 1, 0, 12), suppress_warnings=True),
    UnobservedComponents(level='local linear trend', freq_seasonal=[{'period': 12, 'harmonics': 10}])
]
evaluate_forecaster(fcs)
