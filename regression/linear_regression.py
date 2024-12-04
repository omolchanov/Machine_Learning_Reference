from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv('../assets/headbrain.csv')

# print(df.describe())

# Declaring independent and dependent variables
X = df['Head Size(cm^3)'].values
y = df['Brain Weight(grams)'].values


def build_manual_model():
    # Finding mean of X and y
    X_ = np.mean(X)
    y_ = np.mean(y)

    # Calculating m and c coefficients for the linear regression
    # y = mx + c
    num = 0
    den = 0

    for i in range(len(X)):
        num += (X[i] - X_) * (y[i] - y_)
        den += (X[i] - X_) ** 2

    m = num / den
    c = y_ - (m * X_)

    print('m: ', m, 'C:', c)

    # Plotting initial samples and the linear regression line
    plt.scatter(X, y, c='orange')
    plt.plot(X, m*X+c, c='red')

    plt.xlabel('Head Size(cm^3)')
    plt.ylabel('Brain Weight(grams)')
    plt.show()

    # Calculating R2 score
    ss_t = 0
    ss_r = 0

    for i in range(len(X)):
        y_pred = c + m * X[i]
        ss_t += (y[i] - y_) ** 2
        ss_r += (y[i] - y_pred) ** 2

    r2 = 1 - (ss_r / ss_t)
    print('R2 score: ', r2)


def build_sklearn_model():
    X_r = X.reshape(-1, 1)

    reg = LinearRegression()
    reg.fit(X_r, y)

    m = reg.coef_[0]
    c = reg.intercept_
    print('m: ', m, 'C:', c)

    # Plotting initial samples and the linear regression line
    plt.scatter(X_r, y, c='orange')
    plt.plot(X_r, m * X_r + c, c='red')

    plt.xlabel('Head Size(cm^3)')
    plt.ylabel('Brain Weight(grams)')
    plt.show()

    r2 = reg.score(X_r, y)
    print('R2 score: ', r2)


build_manual_model()
build_sklearn_model()
