from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.datasets import make_regression
from sklearn.ensemble import BaggingRegressor
import numpy as np


# Prepare dataset
X, y = make_regression(n_samples=100, n_features=10, n_informative=10, noise=0.1, random_state=0)

model = BaggingRegressor()

# Eveluate the model: MAE (mean absolute error) and MSE (mean squared error) scorers
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=0)

mae_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=1, error_score='raise')
mse_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=1, error_score=1)


print('MAE mean: ', np.mean(mae_scores))
print('MAE std:', np.std(mae_scores))
print('MSE mean:', np.mean(mse_scores))

# Make prediction
model.fit(X, y)

x_pred = row = [[0.88950817, -0.93540416, 0.08392824, 0.26438806, -0.52828711, -1.21102238, -0.4499934, 1.47392391,
                 -0.19737726, -0.22252503]]

y_pred = model.predict(x_pred)
print('Predicted result: ', y_pred[0])
