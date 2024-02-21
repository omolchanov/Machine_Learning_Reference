# Guideline
# https://medium.com/nuances-of-programming/%D0%BB%D0%B0%D1%81%D1%81%D0%BE-%D0%B8-%D1%80%D0%B8%D0%B4%D0%B6-%D1%80%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%B8%D0%B8-%D0%B8%D0%BD%D1%82%D1%83%D0%B8%D1%82%D0%B8%D0%B2%D0%BD%D0%BE%D0%B5-%D1%81%D1%80%D0%B0%D0%B2%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5-a542dd761f62

from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pandas as pd


col_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOS', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

df = pd.read_csv('../assets/boston_houses.csv', names=col_names, delim_whitespace=True, header=None).head(100)

X = df.drop(['MEDV'], axis=1)
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def linear_regression():
    reg = LinearRegression()

    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    print('MSE: %.3f' % mean_squared_error(y_test, y_pred))


def lasso_regression():
    """
    Lasso regression imposes a penalty on the l1-norm of the beta vector.
    The l1-norm of a vector is the sum of the absolute values in that vector.

    Lasso regression should be used when there are a few characteristics with high predictive ability and the rest are
    useless. It resets useless characteristics and leaves only a subset of variables.
    """

    lambda_values = [0.000001, 0.0001, 0.001, 0.005, 0.01, 0.05,  0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1]

    for l in lambda_values:

        reg = Lasso(alpha=l)

        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        print('Alpha: %.6f MSE: %.5f' % (l, mean_squared_error(y_test, y_pred)))

        # The model resets about half of the coefficients. It kept only 8 of the 14 coefficients, but retained
        # a fairly large weight for one of them, RM, that represents the average number of rooms in a house.
        print(reg.coef_, '\n')


def ridge_regression():
    """
    Ridge regression imposes a penalty on the l2 norm of the beta vector. The 2-norm of a vector is the square root of
    the sum of the squares of the values in the vector.

    Ridge regression is best used when the predictive ability of a data set is distributed among different
    characteristics. Ridge regression does not null out characteristics that can be useful in making forecasts, but
    simply reduces the weight of most variables in the model.
    """

    lambda_values = [0.00001, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 3, 5, 6, 7, 8, 9, 10]

    for l in lambda_values:

        reg = Ridge(alpha=l)

        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        print('Alpha: %.6f MSE: %.5f' % (l, mean_squared_error(y_test, y_pred)))
        print(reg.coef_)


linear_regression()
lasso_regression()
ridge_regression()
