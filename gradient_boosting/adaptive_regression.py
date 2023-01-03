from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.datasets import make_regression

import matplotlib.pyplot as plt

# Prepare the dataset
X, y = make_regression(n_samples=500, n_features=2)

# Plotting initial dataset
plt.figure(figsize=(15, 7))
plt.scatter(X[:, 0], X[:, 1])

plt.title('Initial dataset')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


def adaptive_boosting_regression(estimator_model=None):

    # Fitting the model
    model = AdaBoostRegressor(n_estimators=20, base_estimator=estimator_model)
    model.fit(X_train, y_train)

    print('Model: %s' % model)

    # Calculating R2 score and error for each estimator
    for i, estimator in enumerate(model.estimators_):
        y_pred = estimator.predict(X_test)
        score = round(r2_score(y_test, y_pred), 3)
        error = round(model.estimator_errors_[i], 3)
        weight = round(model.estimator_weights_[i], 3)

        print('Estimator #%s | R2 score: %s | Error: %s | Weight: %s' % (i, score, error, weight))

    # Calculating final R2 score
    y_pred = model.predict(X_test)
    score = round(r2_score(y_test, y_pred), 3)

    print('Final R2 score: %s' % score)


cl_l = LinearRegression()
adaptive_boosting_regression(cl_l)
