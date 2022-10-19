from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import numpy as np


def cluster_stores(plot_centroids=False):
    # Dataset with coordinates of stores
    X = np.array([
        [5, 3], [10, 15], [15, 12], [24, 10], [30, 45], [85, 70],
        [71, 80], [60, 78], [55, 52], [80, 91], [10, 80], [12, 80]
    ])

    plt.scatter(X[:, 0], X[:, 1])
    plt.title('Initial dataset')
    plt.show()

    model = AffinityPropagation(verbose=True)
    y = model.fit(X)

    y_pred = model.predict(X)
    print('Silhouette Score: ', silhouette_score(X, y_pred), '\n')

    plt.scatter(X[:, 0], X[:, 1], c=y.labels_, cmap='rainbow')
    plt.title('Clustering stores with Affinity propagation')

    if plot_centroids is True:
        plt.scatter(y.cluster_centers_[:, 0], y.cluster_centers_[:, 1], c='black')

    plt.show()


cluster_stores()
