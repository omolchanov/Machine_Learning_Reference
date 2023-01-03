from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.datasets import make_regression

import numpy as np

# Prepare the dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)

model = GradientBoostingRegressor()

# Cross-validation
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
scores = cross_val_score(model, X, y, scoring='r2', cv=cv, n_jobs=-1)

print('Mean R2 score: %s' % (np.mean(scores)))

# Fitting and predicting
model.fit(X, y)

X_pred = [
    0.20543991,
    -0.97049844,
    -0.81403429,
    -0.23842689,
    -0.60704084,
    -0.48541492,
    0.53113006,
    2.01834338,
    -0.90745243,
    -1.85859731,
    -1.02334791,
    -0.6877744,
    0.60984819,
    -0.70630121,
    -1.29161497,
    1.32385441,
    1.42150747,
    1.26567231,
    2.56569098,
    -0.11154792
]

y_pred = model.predict([X_pred])
print('Predicted Result: ', y_pred[0])
