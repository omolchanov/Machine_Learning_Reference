import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.inf)

# Read and prepare the dataset
df = pd.read_csv('../assets/churn-rate.csv')

columns_to_float = [
    'total day minutes',
    'total day charge',
    'total eve minutes',
    'total eve charge',
    'total night minutes',
    'total night charge',
    'total intl minutes',
    'total intl charge'
]

for i, name in enumerate(columns_to_float):
    df[name] = df[name].str.replace(",", ".").astype('float64')


def plot_customers_churn():
    """
    Plots a histogram for loyal and churn customers
    """

    df.loc[df['churn'] == False, 'customer service calls'].hist(label='Loyal')
    df.loc[df['churn'] == True, 'customer service calls'].hist(label='Churn')

    plt.xlabel('customer service calls')
    plt.ylabel('quantity')

    plt.legend()
    plt.show()


def get_bootstrap_samples(data, n_samples):
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]

    return samples


def stat_intervals(stat, alpha):
    """Produce an interval estimate."""

    # https://investprofit.info/percentile/
    boundaries = np.percentile(stat, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return boundaries


loyal_customers = df.loc[df['churn'] == False, 'customer service calls'].values
churn_customers = df.loc[df['churn'] == True, 'customer service calls'].values


loyal_customers_mean = [np.mean(s) for s in get_bootstrap_samples(loyal_customers, 1000)]
churn_customers_mean = [np.mean(s) for s in get_bootstrap_samples(churn_customers, 1000)]

print('Service calls from loyal customers: mean interval', stat_intervals(loyal_customers_mean, 0.05))
print('Service calls from churn customers: mean interval', stat_intervals(churn_customers_mean, 0.05))

