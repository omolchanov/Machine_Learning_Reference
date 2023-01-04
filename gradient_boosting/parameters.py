from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.datasets import make_classification

import numpy as np
import matplotlib.pyplot as plt

# Prepare the dataset
X, y = make_classification(n_samples=1000)

# Parameters matrix
n_estimators = [10, 50, 100, 250]
n_subsamples = np.arange(0.1, 1.1, 0.1)
n_features = range(1, 20 + 1)
n_learning_rate = [0.0001, 0.001, 0.01, 0.1, 1.0]
n_max_depth = range(1, 10 + 1)


def explore_parameters(parameters):
    results = []

    for p in parameters:
        model = GradientBoostingClassifier(max_depth=p)

        # Cross-validation
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

        results.append(scores)

        print('Parameter value: %s | Accuracy: %s' % (p, np.round(np.mean(scores), 3)))

    plot_results(results, parameters, 'Accuracy with different n_max_depth')


def plot_results(results, labels, title):
    plt.figure(figsize=(15, 7))

    plt.boxplot(results, labels=labels, showmeans=True)
    plt.title(title)
    plt.show()


def grid_search():

    # Composing parameters grid
    grid = dict()

    grid['n_estimators'] = n_estimators
    grid['subsample'] = n_subsamples
    # grid['max_features'] = n_features

    # Model and Cross-validation
    model = GradientBoostingClassifier()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)

    # Grid Search
    gs = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')
    gs.fit(X, y)

    print('Best parameters: %s | Best score: %s' % (gs.best_params_, gs.best_score_))


explore_parameters(n_max_depth)
grid_search()
