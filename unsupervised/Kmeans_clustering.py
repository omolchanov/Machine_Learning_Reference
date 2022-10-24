from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import numpy as np

# Dataset with coordinates of stores
X = np.array([[5, 3], [10, 15], [15, 12], [24, 10], [30, 45], [85, 70], [71, 80], [60, 78], [55, 52], [80, 91]])


def get_best_n_clusters():
    wcss = []

    for n in range(1, 10 + 1):
        model = KMeans(n_clusters=n, random_state=42)
        model.fit(X)

        # Calculating WCSS. Identifying clusters' accuracy
        # The smaller the WCSS is, the closer our points are, therefore we have a more well-formed cluster
        print('WCSS for K=%s: %s' % (model.n_clusters, model.inertia_))

        print('Groups: ', model.labels_)
        print('Centroids: \n', model.cluster_centers_, '\n')

        wcss.append(model.inertia_)

    ks = np.arange(1, len(wcss) + 1)

    #  The lowest elbow point should be opted, so, it would be 3 in our case
    plt.plot(ks, wcss)

    plt.xlabel('K')
    plt.ylabel('WCSS')
    plt.title('WCSS for each K')

    plt.axvline(3, color='r')
    plt.show()


def build_clusters(plot_centroids=False):
    # Clustering with the best n_clusters
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(X)

    # Calculating Silhouette Coefficient, a metric used to calculate the goodness of a clustering technique.
    # Its value ranges from -1 to 1.
    # 1: Means clusters are well apart from each other and clearly distinguished.
    # 0: Means clusters are indifferent, or we can say that the distance between clusters is not significant.
    # -1: Means clusters are assigned in the wrong way.
    y_pred = model.predict(X)
    print('Silhouette Score: ', silhouette_score(X, y_pred), '\n')

    # Plotting points
    plt.scatter(X[:, 0], X[:, 1], c=[model.labels_])
    plt.title('Clustering with K-Means')

    if plot_centroids is True:
        # Plotting centroids
        plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1])
        plt.axvline(model.cluster_centers_[:, 0][0])
        plt.axvline(model.cluster_centers_[:, 0][1])
        plt.axvline(model.cluster_centers_[:, 0][2])

    plt.show()


def assign_to_cluster(point):
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(X)

    print('Point %s is assigned to cluster %s' % (point, model.predict([point])[0]))


get_best_n_clusters()
build_clusters(plot_centroids=True)
assign_to_cluster([60, 60])
