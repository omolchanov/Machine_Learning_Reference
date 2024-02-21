from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

import pandas as pd

from matplotlib import pyplot as plt

data = {
    'Exp': [2, 2.2, 2.8, 4, 7, 8, 11, 12, 21, 25],
    'Salary': [7, 8, 11, 15, 22, 29, 37, 45.7, 49, 52]
}

df = pd.DataFrame(data)
X = df['Exp'].values.reshape(-1, 1)
y = df['Salary'].values.reshape(-1, 1)


def evaluate_regressor():

    # Tranforms the independent variables. Added columns with exponents of original X basing on degree
    # parameter. E.g. in case of degree=3, two new columns will be added with the tranformed data:
    # Xi ^ 2
    # Xi ^ 3
    # Include bias parameter. If true => adds additional column with value 1.0
    pf = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = pf.fit_transform(X)

    reg = LinearRegression()
    reg.fit(X_poly, y)

    y_pred = reg.predict(X_poly)

    print('MAE: %.3f' % mean_absolute_error(y, y_pred))
    print('MSE: %.3f' % root_mean_squared_error(y, y_pred))
    print('R2 score: %.3f' % r2_score(y, y_pred))

    plt.plot(X, y)
    plt.plot(X, y_pred)

    plt.legend(['y', 'y_pred'])
    plt.xlabel('Expierence')
    plt.ylabel('Salary')
    plt.show()


evaluate_regressor()
