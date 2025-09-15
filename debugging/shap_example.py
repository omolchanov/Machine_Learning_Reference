"""
SHAP (SHapley Additive exPlanations) is a method to explain predictions of machine learning models.
It’s based on Shapley values from cooperative game theory.

Idea:
You have a model (say, predicting house price).
Features: size, number of rooms, district, year built.
Model predicts 100,000$.
SHAP decomposes this prediction into contributions from each feature:

Base value (average prediction): 80,000$
Size adds +15,000$
District adds +8,000$
Floor subtracts −3,000$
Year built adds almost nothing

So: 80,000 + 15,000 + 8,000 − 3,000 = 100,000$

The force plot is one of SHAP’s most famous visualizations.

It explains a single prediction (or multiple predictions if aggregated).
Think of it like a tug-of-war:
Features pushing the prediction higher (red arrows → “positive contribution”).
Features pushing the prediction lower (blue arrows → “negative contribution”).
The length of the arrow shows how strong that feature’s effect is.
"""

import shap
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import numpy as np
np.bool = bool


def regression_example():
    # Sample data
    data = pd.DataFrame({
        "square": [30, 45, 60, 70, 80, 100, 120, 150],
        "rooms":  [1, 2, 2, 3, 3, 4, 4, 5],
        "district": [1, 2, 1, 2, 3, 1, 2, 3],  # 1=center, 2=suburbs, 3=outskirts
        "price":  [40, 60, 90, 110, 130, 170, 200, 240]
    })

    X = data[["square", "rooms", "district"]]
    y = data["price"]

    # Train model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # Visualization
    shap.plots.bar(shap_values)   # Global feature importance
    plt.show()

    # Enable interactive plots
    shap.initjs()

    # Force plot for all samples
    shap.force_plot(explainer.expected_value, shap_values.values, X)
    shap.save_html("regression.html", shap.force_plot(explainer.expected_value, shap_values.values, X))


def classification_example():
    # Load dataset
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)  # shape: (n_samples, n_features, n_classes)

    # Visualizations
    plt.title("Global feature importance for class 0")
    shap.plots.bar(shap_values[:, :, 0])

    plt.title("Global feature importance for class 1")
    shap.plots.bar(shap_values[:, :, 1])

    plt.title("Global feature importance for class 2")
    shap.plots.bar(shap_values[:, :, 2])


def clustering_example():
    # Load dataset
    iris = load_iris(as_frame=True)
    X = iris.data

    # Run clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    # Distances to cluster centers (n_samples × n_clusters)
    distances = kmeans.transform(X)

    # Fit regression model for one cluster distance
    reg = LinearRegression()
    reg.fit(X, distances[:, 0])  # distance to cluster 0

    # SHAP explanation
    explainer = shap.Explainer(reg, X)
    shap_values = explainer(X)

    shap.plots.bar(shap_values)


regression_example()
# classification_example()
# clustering_example()
