# Guideline: https://machinelearningmastery.com/multi-output-regression-models-with-python/

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.svm import LinearSVR

from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

# Generating a dataset on regression problem
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)


def multioutput_regression():
    """
    The direct approach to multioutput regression involves dividing the regression problem
    into a separate problem for each target variable to be predicted.
    """

    regressors = [LinearRegression(), LinearSVR(dual='auto')]

    for reg in regressors:
        wrapper = MultiOutputRegressor(reg)
        wrapper.fit(X, y)

        X_pred = [
            0.21947749, 0.32948997, 0.81560036, 0.440956, -0.0606303, -0.29257894, -0.2820059, -0.00290545,
            0.96402263, 0.04992249
        ]

        y_pred = wrapper.predict([X_pred])

        print('\n', reg)
        print('Predicted: ', y_pred[0])

        y_pred = wrapper.predict(X)
        print('MSE: %.3f' % mean_squared_error(y, y_pred))
        print('R2 Score: %.3f' % r2_score(y, y_pred))


def chain_multioutput_regression():
    """
    Another approach to using single-output regression models for multioutput regression is to create a linear
    sequence of models.

    The first model in the sequence uses the input and predicts one output; the second model uses the input and the
    output from the first model to make a prediction; the third model uses the input and output from the first
    two models to make a prediction, and so on.
    """

    regressors = [LinearRegression(), LinearSVR(dual='auto')]

    for reg in regressors:
        wrapper = RegressorChain(reg)
        wrapper.fit(X, y)

        X_pred = [
            0.21947749, 0.32948997, 0.81560036, 0.440956, -0.0606303, -0.29257894, -0.2820059, -0.00290545,
            0.96402263, 0.04992249
        ]

        y_pred = wrapper.predict([X_pred])

        print('\n', reg)
        print('Predicted: ', y_pred[0])

        y_pred = wrapper.predict(X)
        print('MSE: %.3f' % mean_squared_error(y, y_pred))
        print('R2 Score: %.3f' % r2_score(y, y_pred))


multioutput_regression()
chain_multioutput_regression()
