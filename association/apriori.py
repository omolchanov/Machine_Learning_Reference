# Apriori is an iterative algorithm that works by generating frequent itemsets and using them
# to derive association rules. Apriori uses support and confidence measures to
# identify interesting rules.

# Guideline
# https://www.kaggle.com/code/sangwookchn/association-rule-learning-with-scikit-learn


import sys

from apyori import apriori

import pandas as pd
import numpy as np

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
np.set_printoptions(threshold=sys.maxsize, suppress=True)

df = pd.read_csv('../assets/Market_Basket_Optimisation.csv', header = None)

transactions = []

for i in range(0, df.shape[0]):
    transactions.append([str(df.values[i, j]) for j in range(0, 20)])


rules = apriori(
    transactions,
    min_support=0.003,
    min_confidence=0.2,
    min_lift=3,
    min_length=2
)

results = list(rules)

results = pd.DataFrame(results)
print(results.head(5))
