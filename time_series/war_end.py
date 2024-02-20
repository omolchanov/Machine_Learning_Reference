"""
War_end.py
Time series forecaster predicting the end of war basing on utilizing tanks and APCs.
The data is provided by Ukrainian Ministry of Defence
"""

from sktime.forecasting.naive import NaiveForecaster

from sktime.split import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

from sktime.utils.plotting import plot_series
from matplotlib import pyplot as plt

import pandas as pd

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
pd.set_option('display.float_format', lambda x: '%.i' % x)

plt.rc('font', size=10)

# Source: https://www.kaggle.com/datasets/piterfm/2022-ukraine-russian-war
df = pd.read_csv('../assets/russia_losses_equipment.csv')

# Setting date as index
df.index = pd.PeriodIndex(df['date'], freq='M')

# Extracting and renaming the required columns
df = df[['tank', 'APC']]
df.columns = ['tank_lost', 'APC_lost']

# Finding the delta of montly loses and grouping data by month
df = df.diff().fillna(df).astype(int).groupby('date').sum()

# Setting the data on equipment before invasion and on storage
# Sources:
# https://www.minusrus.com/ru
# https://www.rbc.ua/ukr/news/chi-spravdi-rosiyi-nezlichenni-zapasi-zbroyi-1694512803.html
# https://www.unian.net/weapons/rossiya-imeet-na-hranenii-eshche-okolo-9-tysyach-bbm-analitiki-12432696.html

BEFORE_INVASION = {'tank': 3300, 'APC': 13758}
ON_STORAGE = {'tank': 5000, 'APC': 8917}


def find_correlation():
    """
    Finds the correlation betweem loses of tanks and APCs
    """
    corr = df['tank_lost'].corr(df['APC_lost'])
    print('\nCorrelation: %.3f' % corr)


# Splitting the data
X = pd.DataFrame(df.index)

# Setting the period range for forecasting
start_fh = df.index[-1] + 1
fh = pd.period_range(start=start_fh, freq='M', periods=36)

# Selecting a TS model for forecasting
fc = NaiveForecaster(sp=12)

# Fitting the model and predicting
fc.fit(df, X=X, fh=fh)
df_pred = fc.predict(X=X, fh=fh)

# Merging the dataframe with exisiting data and the dataframe with predictions
df_f = pd.concat([df, df_pred])


def find_end_period(df, eq):
    """
    Calculates remnants of military equipment basing on the monthly loses. The function is stopped when
    negative or zero balance reached.

    :param df: dataframe with exisitng and predicted data
    :param eq: title of the equimpment [tank, APC]
    :return: month when supplements of the particular equipment ran out
    """

    if len(df.columns) > 2:
        df.drop(columns=df.columns[-1],  axis=1,  inplace=True)

    for i in df.index:
        if i == pd.Period('2022-02'):
            df.loc[i, eq + '_remaining'] = BEFORE_INVASION[eq] + ON_STORAGE[eq] - df.loc[i, eq + '_lost']
            continue

        df.loc[i, eq + '_remaining'] = df.loc[i-1, eq + '_remaining'] - df.loc[i, eq + '_lost']

        if df.loc[i, eq + '_remaining'] < 0:
            df = df[:i]
            return df, str(i)


def plot_results(df, end_period, eq):
    """
    Plots loses of the equipment (existing and predicted) and storage balance

    :param df: dataframe with exisitng and predicted data
    :param end_period: month when supplements of the particular equipment ran out
    :param eq: title of the equimpment [tank, APC]
    """

    df.index = df.index.astype('str')
    x = list(df.index.unique())

    plt.plot(x, df[eq + '_lost'])
    plt.plot(x, df[eq + '_remaining'])

    plt.axvline(x=str(start_fh), color='g', linestyle='--')
    plt.axvline(x=end_period, color='r')

    plt.title(['Forecast', eq, fc])
    plt.legend(['Lost', 'Remaining', 'Start of forecasting period', 'End period'])
    plt.xticks(rotation=90)
    plt.show()


def predict():
    """
    Predicts the end period of war basing on balance of military equipment
    :return war_end: the month when the war is predicted to finish
    """

    find_correlation()

    equipment = ['tank', 'APC']
    end_periods = []

    for eq in equipment:
        df_eq, end_period = find_end_period(df_f, eq)
        plot_results(df_eq, end_period, eq)

        print('\n{} | End Period: {}'. format(eq, end_period))
        print(df_eq)

        end_periods.append(end_period)

    war_end = max(end_periods)
    print('\nPREDICTED END OF THE WAR: %s' % war_end)

    return war_end


def evaluate(models):
    """
    Evaluates different TS forecasters using MAE
    :param models: list of TS forecasters
    """

    y = df[['tank_lost', 'APC_lost']]

    y_train, y_test = temporal_train_test_split(y, test_size=6)
    fh = range(1, 6 + 1)

    for m in models:
        print('\nModel: ', m)

        m.fit(y_train, fh=fh)
        y_pred = m.predict()

        mae = mean_absolute_percentage_error(y_test, y_pred, symmetric=False)
        print('MAE: %.3f' % mae)

        for c in y:
            plot_series(
                y_train[c],
                y_test[c], y_pred[c],
                labels=['y_train', 'y_test', 'y_pred'],
                title=[c, m, mae.__round__(3)]
            )
            plt.show()


predict()
# evaluate([NaiveForecaster(sp=12)])
