"""
Exploratory Data Analysis (EDA) refers to the method of studying and exploring
record sets to apprehend  their predominant traits, discover patterns, locate outliers,
and identify relationships between variables.

Guidelines:
https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15
https://www.geeksforgeeks.org/what-is-exploratory-data-analysis/
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
np.set_printoptions(threshold=np.inf, suppress=True)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)

# Reading the dataset
# Choosing the proper separator
df = pd.read_csv('../assets/winequality-white.csv', sep=';')

# Getting an insight on the dataset
# Printing first 10 rows
print('\n Insight: ')
print(df.head(10))

# The total number of rows and columns
print('\n Shape', df.shape)

# The names of the features
print('\n Features: ', df.columns[:-1].values)
print('\n Label: ', df.columns[-1])

# The columns and corresponding data types
print('\n The columns and corresponding data types: ')
df.info()

# The summary stats
print('\n The summary stats: ')
print(df.describe())

# The classes (dependent variable)
print('\n The classes (dependent variable): ', df['quality'].unique())

# The distribution of the classes
print('\n The distribution of the classes: ', df['quality'].value_counts())

# Checking missed values
print('\n Number of missed values: ', df.isnull().sum())
print('\n Number of n/a values: ', df.isna().sum())

# Correlation between the features
sns.heatmap(df.corr(method='pearson'), annot=True).set(
    title='Correlation between the features'
)

# Whiskey plot. Catching outliers
fig, ax = plt.subplots(1, df.shape[1], sharex=False)

for i, c in enumerate(df.columns):
    sns.boxplot(y=df[c], ax=ax[i])

plt.subplots_adjust(wspace=1.75)

# Distribution of dependent and independet variables
# https://habr.com/ru/companies/skillfactory/articles/683738/
fig, ax = plt.subplots(1, 4, sharey=False)

for i, c in enumerate(df.columns[:4]):
    sns.histplot(df[c], kde=True, ax=ax[i]).set(ylabel='')

plt.show()

# Different binwidth can be applied
sns.histplot(df.iloc[:, -1], kde=True, bins=3)
plt.show()
