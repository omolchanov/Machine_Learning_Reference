import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Configure Pandas
pd.set_option("display.precision", 2)
pd.set_option("expand_frame_repr", False)
pd.set_option('display.max_rows', None)

df = pd.read_csv('assets/churn-rate.csv')

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

# Visualization of unvariative variables
# features = ['total day calls', 'total eve calls', 'total night calls', 'total intl calls']
# print(df[features].describe())

# df[features].hist(figsize=(10, 4))
# df[features].plot(kind="density", subplots=True, layout=(2, 2), sharex=False, figsize=(10, 4))
# sns.distplot(df['total day calls'])

# sns.boxplot(x='total intl calls', data=df)
# sns.violinplot(x='total intl calls', data=df)

# sns.countplot(x='customer service calls', data=df[(df['customer service calls'] > 0)])
# plt.show()

# Correlation matrix
# df.drop(['state', 'area code', 'international plan', 'voice mail plan', 'phone number', 'churn'], axis=1, inplace=True)

# Removing dependent variables
# df.drop(['total day charge', 'total eve charge', 'total night charge', 'total intl charge'], axis=1, inplace=True)
# sns.heatmap(df.corr())

# Scatter plots
# sns.jointplot(df["total day minutes"], df["total intl minutes"], kind="scatter")
# sns.jointplot(df["total day minutes"], df["total night minutes"], kind="kde")

# sns.pairplot(df)
# plt.show(block=True)

# Lm Plot
# sns.lmplot('total day charge', 'total night charge', data=df, hue='churn', fit_reg=False)
# plt.show(block=True)

# Box plots (numerical and categorial variables)
# fig, axs = plt.subplots(ncols=3)
# sns.boxplot(x='churn', y='customer service calls', data=df, ax=axs[0])
# sns.boxplot(x='churn', y='total day minutes', data=df, ax=axs[1])
# sns.boxplot(x='churn', y='number vmail messages', data=df, ax=axs[2])

# sns.catplot(
#     x="churn",
#     y="total day minutes",
#     col="customer service calls",
#     data=df[df["customer service calls"] < 8],
#     kind="box",
#     col_wrap=4,
#     height=3
# )

# Numerical variable vs categorial variable.
# fig, axs = plt.subplots(ncols=3)
# sns.countplot(x='customer service calls', hue='churn', data=df, ax=axs[0])
# sns.countplot(x='international plan', hue='churn', data=df, ax=axs[1])
# sns.countplot(x='voice mail plan', hue='churn', data=df, ax=axs[2])
# plt.show()

# Contingency table
# print(pd.crosstab(df['state'], df['churn']).T)
# print(df.groupby(['state'])['churn'].agg([np.mean]).sort_values(by='mean', ascending=False).T)

# t-SNE
X = df.drop(['churn', 'state', 'phone number'], axis=1)
X['international plan'] = X['international plan'].map({'yes': 1, 'no': 0})
X['voice mail plan'] = X['voice mail plan'].map({'yes': 1, 'no': 0})

# Normalizing data. For this, we will subtract the mean from each variable and divide it by its standard deviation.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

tsne = TSNE(random_state=17)
tsne_repr = tsne.fit_transform(X_scaled)

# Visualzing the whole dataset
plt.scatter(
    tsne_repr[:, 0],
    tsne_repr[:, 1],
    c=df['voice mail plan'].map({'no': "blue", 'yes': "orange"}),
    alpha=0.5
)

plt.show()

