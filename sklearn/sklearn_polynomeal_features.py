import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures

# The “degree” of the polynomial is used to control the number of features added, e.g. a degree of 3 will add two new
# variables for each input variable.
DEGREE = 7

# The bias (the value of 1.0)
FEATURE_BIAS = True

# The larger the parameter , the more complex the relationships in the data that the model can recover
# (intuitively  corresponds to the “complexity” of the model - model capacity)
C = 1e4

# Reading the dataset. Splitting into features and labels
df = pd.read_csv('../assets/microchips_dataset.csv', header=None, names=('test1', 'test2', 'released'))

X = df.iloc[:, :2].values
y = df['released']


# Plotting initial classification for the dataset
# Plots boundary between released and faulty chips
def plot_boundary(grid_step=0.005, poly_featurizer=None):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step), np.arange(y_min, y_max, grid_step))

    z = model.predict(poly_featurizer.transform(np.c_[xx.ravel(), yy.ravel()]))
    z = z.reshape(xx.shape)

    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="blue", label="Released")
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="orange", label="Faulty")
    plt.contour(xx, yy, z, cmap='Accent')

    plt.xlabel('test 1')
    plt.ylabel('test 2')

    plt.legend()
    plt.title(
        'Params: Degree: %s | C: %s | Accuracy: %s' % (DEGREE, C, model.score(X_poly, y).__round__(2)))
    plt.show()


# Identifying the optimal value of the regularization parameter C
def identify_optimal_C():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
    c_values = np.logspace(-2, 3, 500)
    model_searcher = LogisticRegressionCV(Cs=c_values, cv=skf, n_jobs=1)
    model_searcher.fit(X_poly, y)

    print('Optimal C value: ', model_searcher.C_[0])

    # Plotting results of the cross-validation
    plt.plot(c_values, np.mean(model_searcher.scores_[1], axis=0))
    plt.xlabel("C")
    plt.ylabel("Mean CV-accuracy")
    plt.show()


# https://machinelearningmastery.com/polynomial-features-transforms-for-machine-learning/
# Polynomial features are those features created by raising existing features to an exponent.
pf = PolynomialFeatures(degree=DEGREE, include_bias=FEATURE_BIAS)
X_poly = pf.fit_transform(X)

model = LogisticRegression(C=C)
model.fit(X_poly, y)

plot_boundary(poly_featurizer=pf)
identify_optimal_C()
