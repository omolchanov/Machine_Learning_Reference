import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# Read and prepare the dataset
df = pd.read_csv('../assets/churn-rate.csv')

df['international plan'] = df['international plan'].map({'yes': 1, 'no': 0})
df['voice mail plan'] = df['voice mail plan'].map({'yes': 1, 'no': 0})
df['churn'] = df['churn'].map({True: 1, False: 0})

columns_to_float = [
    'total day minutes',
    'total day charge',
    'total eve minutes',
    'total eve charge',
    'total night minutes',
    'total night charge',
    'total intl minutes',
    'total intl charge'
]

for i, name in enumerate(columns_to_float):
    df[name] = df[name].str.replace(",", ".").astype('float64')


X = df.drop(['churn', 'account length', 'area code', 'state', 'phone number'], axis=1).values
y = df['churn'].values


def plot_curves_with_error(x, data, label):
    """
    Plots validation/learning curves with error.
    Cross-validation error shows how well the model fits the data (the existing trend in the data) while retaining the
    ability to generalize to new data
    """

    mu, std = data.mean(1), data.std(1)
    lines = plt.plot(x, mu, label=label)

    plt.fill_between(
        x,
        mu - std,
        mu + std,
        edgecolor="none",
        facecolor=lines[0].get_color(),
        alpha=0.2,
    )


def plot_validation_curves():
    """
    Calculates and plots validation curves for train and test sets. Validation curve is a graph showing the results on
    training and validation sets depending on the complexity of the model.

    if the two curves are close to each other and both errors are large, it is a sign of underfitting
    if the two curves are far from each other, it is a sign of overfitting
    """

    # Train logistic regression with stochastic gradient descent
    # https://towardsdatascience.com/stochastic-gradient-descent-clearly-explained-53d239905d31
    alphas = np.logspace(-2, 0, 20)

    clf = SGDClassifier(loss='log', n_jobs=1, random_state=17, max_iter=1)
    pipeline = Pipeline(
        [
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2)),
            ('sgd_logit', clf),
        ]
    )

    # https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/
    score_train, score_test = validation_curve(
        estimator=pipeline, X=X, y=y, param_name='sgd_logit__alpha', param_range=alphas, cv=5, scoring='roc_auc'
    )

    plot_curves_with_error(alphas, score_train, label='Training scores')
    plot_curves_with_error(alphas, score_test, label='Test scores')

    plt.xlabel('alpha')
    plt.ylabel("ROC AUC")

    plt.title('Validation Curves')
    plt.legend()
    plt.show()


def plot_learning_curves(degree=2, alpha=1e-4):
    """
    Calculates and plots learning curves for train and test sets.Learning Curve is a graph showing the results on
    training and test sets depending on the number of observations.

    if the curves converge, adding new data won’t help, and it is necessary to change the complexity of the model
    if the curves have not converged, adding new data can improve the result
    """

    train_sizes = np.linspace(0.05, 1, 20)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree)),
            (
                "sgd_logit",
                SGDClassifier(n_jobs=1, random_state=17, alpha=alpha, max_iter=5),
            ),
        ]
    )

    size_train_set, score_train, score_test = learning_curve(
        pipeline, X, y, train_sizes=train_sizes, cv=5, scoring="roc_auc"
    )

    plot_curves_with_error(size_train_set, score_train, label='Training scores')
    plot_curves_with_error(size_train_set, score_test, label='Test scores')

    plt.xlabel('Train Set size')
    plt.ylabel("ROC AUC")

    plt.title('Learning Curves')
    plt.legend()
    plt.show()


plot_validation_curves()
plot_learning_curves()
