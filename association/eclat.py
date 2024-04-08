# Eclat (Equivalence Class Transformation) is algorithm for association rule learning.
# It efficiently finds frequent itemsets using a depth-first search strategy, where transaction lists
# are intersected to identify common items.
# Eclat is particularly effective when dealing with vertical datasets.

# Guideline:
# https://hands-on.cloud/implementation-of-eclat-algorithm-using-python/

import sys

from pyECLAT import ECLAT, Example2

import pandas as pd
import numpy as np

import plotly.express as px

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
np.set_printoptions(threshold=sys.maxsize, suppress=True)

# Loading the dataset
df = Example2().get()
# print(df.describe())

eclat = ECLAT(data=df)

# Sparse matrix of products/transactions
# print(eclat.df_bin.head(2))

# Count items in each column (per product)
items_total = eclat.df_bin.astype(int).sum(axis=0)
# print(items_total)

# Count items in each row (per transaction)
transaction_items = eclat.df_bin.astype(int).sum(axis=1)
# print(transaction_items)

# Visualizing items distribution
df = pd.DataFrame({
    'items': items_total.index,
    'transactions': items_total.values
}).sort_values('transactions', ascending=False)

# Visualization
df['all'] = 'Tree Map'

fig = px.treemap(
    df.head(50),
    path=['all', 'items'],
    values='transactions',
    color=df['transactions'].head(50),
    hover_data=['items'],
    color_continuous_scale='Blues',
)

# fig.show()

# Generating association rules
# the item shoud appear at least at 3% of transactions
min_support = 3/100

# start from transactions containing at least 2 items
min_combination = 2

# up to maximum items per transaction
max_combination = 3

rule_indices, rule_support = eclat.fit(
    min_support=min_support,
    min_combination=min_combination,
    max_combination=max_combination,
    separator=' & ',
    verbose=True
)

# Association rules DataFrame
result = (pd.DataFrame(rule_support.items(), columns=['Items', 'Support']).
          sort_values(by=['Support'], ascending=False))

print(result)
