from mlxtend.frequent_patterns import apriori, association_rules


import pandas as pd

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)

df = pd.read_csv('../assets/OnlineRetail.csv')


# Cleaning up the data
df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype(str)
df = df[~df['InvoiceNo'].str.contains('C')]

# Selecting invoices from France only
basket = (df[df['Country'] == 'France'].
          groupby(['InvoiceNo', 'Description'])['Quantity'].
          sum().unstack().reset_index().fillna(0).
          set_index('InvoiceNo'))


def encode_units(x):
    x = 0 if x <= 0 else 1
    return x


basket_sets = basket.map(encode_units)

frequent_itemsets = apriori(basket_sets, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
print(rules)
