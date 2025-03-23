# https://www.geeksforgeeks.org/local-outlier-factor/
# https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html
# https://medium.com/@ilyurek/anomaly-detection-with-local-outlier-factor-lof-b1b82227c15e

import sys

from sklearn.neighbors import LocalOutlierFactor

import pandas as pd
import numpy as np

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
np.set_printoptions(threshold=sys.maxsize, suppress=True)

df = pd.read_csv('../assets/churn-rate.csv', index_col='phone_number')
df = df.drop(['area_code'], axis=1)

X = df.select_dtypes(include=['float', 'int']).columns

clf = LocalOutlierFactor(n_neighbors=5)

# Outlier label. -1 stands for outliers, 1 for inliers
y_pred = clf.fit_predict(df[X])
print(pd.Series(y_pred).value_counts())

# The higher, the more normal. Inliers tend to have a LOF score close to 1 (negative_outlier_factor_ close to -1),
# while outliers tend to have a larger LOF score.
lof = clf.negative_outlier_factor_
print(lof)
