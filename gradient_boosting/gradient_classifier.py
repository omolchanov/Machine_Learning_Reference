from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.datasets import make_classification

import numpy as np

# Prepare the dataset
X, y = make_classification(n_samples=1000)

model = GradientBoostingClassifier(n_estimators=10)

# Cross-validation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

print('Mean accuracy: %s | Mean deviation: %s' % (np.mean(scores), np.std(scores)))

# Fitting and predicting
model.fit(X, y)

X_pred = [
    0.2929949,
    -4.21223056,
    -1.288332,
    -2.17849815,
    -0.64527665,
    2.58097719,
    0.28422388,
    -7.1827928,
    -1.91211104,
    2.73729512,
    0.81395695,
    3.96973717,
    -2.66939799,
    3.34692332,
    4.19791821,
    0.99990998,
    -0.30201875,
    -4.43170633,
    -2.82646737,
    0.44916808
]

y_pred = model.predict([X_pred])
print('Predicted class: ', y_pred[0])
