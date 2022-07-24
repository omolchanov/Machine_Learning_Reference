import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.tree._export import export_text
from sklearn.metrics import mean_squared_error


# Abstract visualiztion of entropy and Gini coef (binary classification)
plt.figure(figsize=(10, 10))

xx = np.linspace(0, 1, 50)
# print(xx)

# plt.plot(xx, [2 * x * (1 - x) for x in xx], label="gini")

# Entropy can be described as the degree of chaos in the system.
# The higher the entropy, the less ordered the system and vice versa.
# plt.plot(xx, [-x * np.log2(x) - (1 - x) * np.log2(1 - x) for x in xx], label="entropy")

# misclassification rate is a metric that tells us the percentage of observations that were incorrectly predicted by
# a classification model.
# Misclassification Rate = # incorrect predictions / # total predictions
# 0 represents a model that had zero incorrect predictions.
# 1 represents a model that had completely incorrect predictions.
# plt.plot(xx, [1 - max(x, 1 - x) for x in xx], label="missclass")

# plt.xlabel('p+ (probability of an object having a label +)')
# plt.ylabel("criteria")
# plt.title("Criteria of quality as a function of p+ (binary classification)")
# plt.legend()

# Generating the dataset for classification
np.random.seed(0)

X = np.random.normal(size=(100, 2))
y = np.random.choice([0, 1], 100)

# sns.scatterplot(X[:, 0], X[:, 1], hue=y, palette='dark')


# There is line shadow showing the confidence interval, because the dataset contains multiple y values for each x value
# sns.regplot(X[:, 0], X[:, 1], scatter=False)

# Descision Tree classification
# model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=17, min_samples_leaf=1)
# model.fit(X, y)

# x_pred = np.random.normal(size=(100, 2))
# y_pred = model.predict(x_pred)

# sns.scatterplot(x_pred[:, 0], x_pred[:, 1], hue=y_pred)
# sns.regplot(x_pred[:, 0], x_pred[:, 1], scatter=False)

# plt.tight_layout()
# plt.show()

# Descision tree with numerical features
data = pd.DataFrame(
    {
        "Age": [17, 64, 18, 20, 38, 49, 55, 25, 29, 31, 33],
        "Salary": [25, 80, 22, 36, 37, 59, 74, 70, 33, 102, 88],
        "Loan Default": [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
    }
)

# model = DecisionTreeClassifier(random_state=17, max_depth=3)
# model.fit(data[['Age', 'Salary']].values, data['Loan Default'].values)

# Print descision path
# tree_rules = export_text(model, feature_names=['Age', 'Salary'], )
# print(tree_rules)


# Descision Tree Regressor
# https://www.geeksforgeeks.org/python-decision-tree-regression-using-sklearn/
N_TRAIN = 20
N_TEST = 10
NOISE = 0.1


def f(x):
    x = x.ravel()
    return np.exp(-(x ** 2)) + 1.5 * np.exp(-((x - 2) ** 2))


def generate(n_samples, noise):
    X = np.random.rand(n_samples) * 10 - 5
    X = np.sort(X).ravel()

    y = (
            np.exp(-(X ** 2))
            + 1.5 * np.exp(-((X - 2) ** 2))
            + np.random.normal(0.0, noise, n_samples)
    )

    X = X.reshape((n_samples, 1))
    return X, y


X_train, y_train = generate(N_TRAIN, NOISE)
X_test, y_test = generate(N_TEST, NOISE)

model = DecisionTreeRegressor(max_depth=5, random_state=17)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

plt.scatter(X_train, y_train, c='b')
plt.scatter(X_test, y_pred, c='r')

plt.plot(X_test, f(X_test), c='m')
plt.plot(X_test, y_pred, c='g')

plt.title('Descision Tree Regressor. MSE=%.2f' % mse)
plt.show()
