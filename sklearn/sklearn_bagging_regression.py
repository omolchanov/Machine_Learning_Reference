from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor

import numpy as np


# Prepare dataset
X, y = make_regression(n_samples=1150, n_features=10, n_informative=10, noise=0.1, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, random_state=0)


def predict_bagging_sample():
    # Prepare dataset
    X, y = make_regression(n_samples=10000, n_features=10, n_informative=10, noise=0.1, random_state=0)

    model = BaggingRegressor(n_estimators=10)

    # Eveluate the model: MAE (mean absolute error) and MSE (mean squared error) scorers
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)

    mae_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=1, error_score='raise')
    mse_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=1, error_score=1)

    print('MAE mean: ', np.mean(mae_scores))
    print('MAE std:', np.std(mae_scores))
    print('MSE mean:', np.mean(mse_scores))

    # Make prediction
    model.fit(X, y)

    x_pred = [[0.88950817, -0.93540416, 0.08392824, 0.26438806, -0.52828711, -1.21102238, -0.4499934, 1.47392391,
               -0.19737726, -0.22252503]]

    y_pred = model.predict(x_pred)
    print('Predicted result: ', y_pred[0])


def f(x):
    x = x.ravel()
    return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)


def evaluate_single_tree_regressor():
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae_model = mean_absolute_error(y_pred, y_test)
    mse_model = mean_squared_error(y_pred, y_test)

    print('Single tree regressor MAE: ', mae_model)
    print('Single tree regressor MSE: ', mse_model)

    return mae_model, mse_model


def evaluate_bagging_tree_regressor():
    model_tree = DecisionTreeRegressor()

    model = BaggingRegressor(model_tree)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae_model = mean_absolute_error(y_pred, y_test)
    mse_model = mean_squared_error(y_pred, y_test)

    print('Bagging tree regressor MAE: ', mae_model)
    print('Bagging tree regressor MSE: ', mse_model)

    return mae_model, mse_model


def evaluate_random_forest_regressor():
    model = RandomForestRegressor(n_estimators=10)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae_model = mean_absolute_error(y_pred, y_test)
    mse_model = mean_squared_error(y_pred, y_test)

    print('Random forest regressor MAE: ', mae_model)
    print('Random forest regressor MSE: ', mse_model)

    return mae_model, mse_model


evaluate_single_tree_regressor()
evaluate_bagging_tree_regressor()
evaluate_random_forest_regressor()
