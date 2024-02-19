from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.tbats import TBATS

from matplotlib import pyplot as plt

import pandas as pd

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
pd.set_option('display.float_format', lambda x: '%.i' % x)

plt.rc('font', size=10)

df = pd.read_csv('../assets/russia_losses_equipment.csv')

# Setting date as index
df.index = pd.PeriodIndex(df['date'], freq='M')

# Extracting and renaming the required columns
df = df[['tank', 'APC']]
df.columns = ['tank_lost', 'APC_lost']

# Finding the delta and grouping data by month
df = df.diff().fillna(df).astype(int).groupby('date').sum()

# Setting the data on equipment before invasion and on storage
BEFORE_INVASION = {'tank': 3300, 'APC': 15000}
ON_STORAGE = {'tank': 4000, 'APC': 1500}


def find_correlation():
    corr = df['tank_lost'].corr(df['APC_lost'])
    print(corr)


X = pd.DataFrame(df.index)

start_fh = df.index[-1] + 1
fh = pd.period_range(start=start_fh, freq='M', periods=18)

fc = TBATS(sp=12)

fc.fit(df, X=X, fh=fh)
df_pred = fc.predict(X=X, fh=fh)

df = pd.concat([df, df_pred])


def find_end_period(df, eq):
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
    equipment = ['tank', 'APC']

    for eq in equipment:
        df_eq, end_period = find_end_period(df, eq)
        plot_results(df_eq, end_period, eq)

        print('\nEnd Period: ', end_period)
        print(df_eq)


predict()
