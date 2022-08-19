from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import tree

import matplotlib.pyplot as plt
import pandas as pd


# https://towardsdatascience.com/the-mathematics-of-decision-trees-random-forest-and-feature-importance-in-scikit-learn-and-spark-f2861df67e3
def iris_dataset_sample():

    # Load and prepare Iris dataset
    dataset = load_iris()
    X = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])
    y = pd.Series(dataset['target']).map({0: 0, 1: 0, 2: 1})

    # Fit Random Forest model
    model = RandomForestClassifier(criterion='gini', n_estimators=3, max_depth=3, random_state=17)
    model.fit(X, y)

    tree_list = model.estimators_

    # Plot a descision graph for each tree in Random Forest
    for i, k in enumerate(tree_list):
        plt.figure(figsize=(16, 12))

        tree.plot_tree(
            tree_list[i],
            filled=True,
            feature_names=dataset['feature_names'],
            class_names=['Y', 'N'],
            node_ids=True,
        )

        plt.show()

    # Calculate feature importance for Random Forest
    print('Feature importance: ', X.columns.values, model.feature_importances_)


# Computes features ordered by importance and plots the results
def hostel_features_importance():
    df = pd.read_csv('../assets/hostel_factors.csv')

    X = df.drop(['hostel', 'rating'], axis=1)
    y = df['rating']

    model = RandomForestRegressor(n_estimators=10, max_features=10, random_state=0)
    model.fit(X, y)

    importances = dict(zip(X.columns.tolist(), model.feature_importances_.round(2)))
    print('The most important feature: ', max(importances, key=importances.get))

    # Plot feature importance on the bar graph
    f = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
    print(f)

    plt.bar(f.keys(), f.values())

    plt.title('Feature Importance')
    plt.xlabel('Feature')
    plt.ylabel('Importance score')

    plt.show()


hostel_features_importance()
