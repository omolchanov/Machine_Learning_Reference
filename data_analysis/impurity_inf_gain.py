# https://youtu.be/-ETQ97mXXF0?t=3865

import pandas as pd
import numpy as np

import pprint

from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_csv('../assets/churn-rate.csv', thousands='.', decimal=',')

# print(df.describe())

X = df.drop(['churn'], axis=1)
y = df['churn']

# Entropy calculation
y_entropy = entropy(np.bincount(y), base=2)
print('Entropy: ', y_entropy)

# Information gain calculation
X = X.drop(['state', 'area code', 'phone number', 'international plan', 'voice mail plan'], axis=1)
ig = mutual_info_classif(X, y)

ig_dict = {}
for i in range(len(X.columns)):
    ig_dict[X.columns[i]] = round(ig[i], 3)

ig_dict_sorted = dict(sorted(ig_dict.items(), key=lambda item: item[1], reverse=True))
print('\nInformation gain for each feature: ')
pprint.pp(ig_dict_sorted)
