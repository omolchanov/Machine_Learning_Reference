import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Configure Pandas
pd.set_option("display.precision", 2)
pd.set_option("expand_frame_repr", False)
pd.set_option('display.max_rows', None)

df = pd.read_csv('assets/churn-rate.csv')

# Print headings and first 5 rows
print(df.head())

# Print the shape of the dataset: number of rows and columns
print('Dataset shape: ', df.shape)

# Get headings names
print(df.columns)

# Print headings, five first and last rows, dataset shape
print(df.info)

# Print count of NOT Null columns and data type for each column
print(df.info())

# Change the column (feature) type
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

df['area code'] = df['area code'].astype('object')

print(df.info())

# The describe method shows basic statistical characteristics of each numerical feature (int64 and float64 types)
print(df.describe())
print(df.describe().shape)

# Describe non-numerical features
print(df.describe(include=['object', 'bool']))
print(df.describe(include=['object', 'bool']).shape)

# Count values for column
print(df['churn'].value_counts(normalize=True))
print(df['state'].value_counts())

# Sort values by a particular column
print(df.sort_values(by=['total day charge'], ascending=False))

# Retrieve data for a single column
print(df["churn"].mean())

# Fetch aggregated data
print(df[df['churn'] == 1]['total day minutes'].mean())
print(df[df['churn'] == 1]['international plan'].count())

query = df[(df['churn'] == 1) & (df['international plan'] == 'yes')]['total intl minutes'].mean()
print(query)

# Indexing the dataset
print(df.iloc[-1:, 0:3])

# Applying functions to the dataframe
print(df.apply(np.max))

result = df['total day calls'].apply(lambda calls: calls * 2)
print(result.iloc[0:20])

# Replace values in a column
churn_dict = {False: 0, True: 1}
df['tr_churn'] = df['churn'].map(churn_dict)
print(df['tr_churn'].iloc[0:30])

# Grouping
columns = ['account length', 'total eve calls']
result = df.groupby(['state', 'churn', 'international plan'])[columns]
print(result)

# Summary tables
print(pd.crosstab(df['churn'], df['international plan'], normalize=True))

# Pivots
pivot = df.pivot_table(['account length', 'total eve calls'], ['state'], aggfunc='mean')
print(pivot)

# Modyfing the dataframe
df['total charge'] = \
    df['total day charge'] + \
    df['total eve charge'] + \
    df['total night charge'] + \
    df['total intl charge']

print(df['total charge'].head())

df.drop(['area code'], axis=1, inplace=True)
df1 = df.drop(df.index[:3330])
print(df1)

# Visualization
ct = pd.crosstab(df['churn'], df['international plan'], margins=True)
print(ct)
# sns.countplot(x='international plan', hue='churn', data=df).set(title='International Plan vs Churn')

ct = pd.crosstab(df['churn'], df['customer service calls'], margins=True)
print(ct)
# sns.countplot(x='customer service calls', hue='churn', data=df).set(title='Customer calls vs Churn')

df['calls_boundary'] = (df['customer service calls'] > 3)
ct = pd.crosstab(df['calls_boundary'], df['churn'], margins=True)
print(ct)
sns.countplot(x='calls_boundary', hue='churn', data=df).set(title='Customer calls boundary vs Churn')

ct = pd.crosstab(df['calls_boundary'] & df['international plan'], df['churn'], margins=True, normalize=True)
print(ct)

plt.show()
