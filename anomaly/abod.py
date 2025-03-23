# https://blog.paperspace.com/outlier-detection-with-abod/

from pyod.utils.data import generate_data
from pyod.models.abod import ABOD
from sklearn.metrics import mean_absolute_error

import pandas as pd

import matplotlib.pyplot as plt

X_train, y_train = generate_data(n_train=150, n_features=2, train_only=True, contamination=0.1, random_state=42)
x1, x2 = X_train[:,0], X_train[:,1]

plt.scatter(x1, x2, c=y_train)
plt.title('Synthetic Data with Outliers')
plt.show()

clf = ABOD(contamination=0.1, method='fast', n_neighbors=10)
clf.fit(X_train)

y_pred = clf.predict(X_train)
print(pd.Series(y_pred).value_counts())

print('MAE: %.3f' % (mean_absolute_error(y_pred, y_train) * 100))

plt.scatter(x1, x2, c=y_pred)
plt.title('ABOD predictions for Outlier Detection')
plt.show()
