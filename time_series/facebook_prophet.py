from sklearn.model_selection import train_test_split

import pandas as pd

import matplotlib.pyplot as plt


# Configure Pandas
pd.set_option("display.precision", 2)
pd.set_option("expand_frame_repr", False)
pd.set_option('display.max_rows', None)

# The dataset
df = pd.read_csv('../assets/medium_posts.csv', on_bad_lines='skip', encoding='utf-8', sep='\t')

df = df[['published', 'url']].dropna().drop_duplicates()
df['published'] = pd.to_datetime(df['published'])

# Remove outliers
df = df[(df['published'] > '2012-08-15') & (df['published'] < '2017-06-26')].sort_values(by=['published'])

# Aggregating and counting unique posts at each given point in time.
aggr_df = df.groupby('published')[['url']].count()
aggr_df.columns = ['posts']

# Resampling time index down to 1-day bins
daily_df = aggr_df.resample('D').apply(sum)

# Resampling time index down to 1-week bins
weekly_df = daily_df.resample('W').apply(sum)

# Omitting the first few years of observations, up to 2015
daily_df = daily_df.loc[daily_df.index >= '2015-01-01']

# Preparing the dataset for Prophet
df = daily_df.reset_index()
df.columns = ['ds', 'y']
df['ds'] = df['ds'].dt.tz_convert(None)

X = df['ds']
y = df['y']


# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


def plot_dataset(data, title):
    plt.figure(figsize=(15, 7))

    plt.plot(data)
    plt.title(title)
    plt.show()


plot_dataset(daily_df['posts'], 'Daily Posts')
plot_dataset(weekly_df['posts'], 'Weekly Posts')
