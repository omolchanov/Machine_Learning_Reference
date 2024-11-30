import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_csv('../assets/churn-rate.csv', thousands='.', decimal=',')
df_cont = df.drop(['state', 'area code', 'phone number', 'international plan', 'voice mail plan', 'churn'], axis=1)

print(df_cont.describe())


def plot_kde():
    for i,c in enumerate(df_cont.columns):
        sns.kdeplot(df_cont[c])

        plt.title(c)
        plt.show()


plot_kde()
