import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.array([
    [6.11, 17.59],
    [5.52, 9.13],
    [8.51, 13.66],
    [7.0, 11.85],
    [5.85, 6.82]])

df = pd.DataFrame(data, columns=['Population', 'Profit'])

# Visuliazing training data
# df.plot(kind='scatter', x='Population', y='Profit', figsize=(8, 8))
# plt.show()


df.insert(0, 'Theta0', 1)
cols = df.shape[1]

X = df.iloc[:, 0:cols-1]
Y = df.iloc[:, cols-1:cols]

theta = np.matrix(np.array([0] * X.shape[1]))
X = np.matrix(X.values)
Y = np.matrix(Y.values)


# Determination coefficient calculation
# https://craftappmobile.com/coefficient-of-determination/#__R
def calculate_rss(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


def gradient_descent(X, Y, theta, alpha, iters):
    t = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - Y
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            t[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))
            theta = t
    cost[i] = calculate_rss(X, Y, theta)
    return theta, cost


# Calculating error without applying of gradient descent
error = calculate_rss(X, Y, theta)
print('Error without applying of gradient descent: ', error)

# Calculating gradient descent
g, cost = gradient_descent(X, Y, theta, 0.01, 1000)

# Calculating error using gradient descent
error = calculate_rss(X, Y, g)
print('Error using gradient descent: ', error)

# Visualizing prediction with gradient descent
x = np.linspace(df.Population.min(), df.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)
fig, ax = plt.subplots(figsize=(12,8))

ax.plot(x, f, 'r', label='Prediction')
ax.scatter(df.Population, df.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

plt.show()
