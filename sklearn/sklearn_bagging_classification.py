import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.datasets import make_circles
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

X, y = make_circles(n_samples=500, factor=0.1, noise=0.35, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

x_pred = np.random.uniform(X.min(), X.max(), [50, 2])


def evaluate_single_tree_classifier():
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    model_as = accuracy_score(y_pred, y_test)
    model_mae = mean_absolute_error(y_pred, y_test)

    y_pred = model.predict(x_pred)

    xx = np.concatenate((X, x_pred))
    yy = np.concatenate((y, y_pred))

    sns.scatterplot(xx[:, 0], xx[:, 1], c=yy)

    plt.title('Single Tree AS: %s | Single Tree MAE: %s' % (model_as, model_mae))
    plt.show()


def evaluate_bagging_tree_classifier():
    model_tree = DecisionTreeClassifier()

    model = BaggingClassifier(model_tree, n_estimators=300, random_state=0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    model_as = accuracy_score(y_pred, y_test)
    model_mae = mean_absolute_error(y_pred, y_test)

    y_pred = model.predict(x_pred)

    xx = np.concatenate((X, x_pred))
    yy = np.concatenate((y, y_pred))

    sns.scatterplot(xx[:, 0], xx[:, 1], c=yy)

    plt.title('Bagging Tree AS: %s | Bagging Tree MAE: %s' % (model_as, model_mae))
    plt.show()


def evaluate_random_forest_classifier():

    # n_estimators — the number of trees in the forest (default = 10)
    model = RandomForestClassifier(n_estimators=300, random_state=0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    model_as = accuracy_score(y_pred, y_test)
    model_mae = mean_absolute_error(y_pred, y_test)

    y_pred = model.predict(x_pred)

    xx = np.concatenate((X, x_pred))
    yy = np.concatenate((y, y_pred))

    sns.scatterplot(xx[:, 0], xx[:, 1], c=yy)

    plt.title('Random Forest AS: %s | Random Forest MAE: %s' % (model_as, model_mae))
    plt.show()


evaluate_single_tree_classifier()
evaluate_bagging_tree_classifier()
evaluate_random_forest_classifier()
