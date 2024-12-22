"""
Fuzzy Clustering is a type of clustering algorithm in machine learning that allows a data point to belong to more than
one cluster with different degrees of membership. Unlike traditional clustering algorithms, such as k-means or
hierarchical clustering, which assign each data point to a single cluster, fuzzy clustering assigns a membership degree
between 0 and 1 for each data point for each cluster.

https://www.geeksforgeeks.org/ml-fuzzy-clustering/
"""

import numpy as np
import skfuzzy as fuzz

import matplotlib.pyplot as plt


X = np.array([
    [5, 3],
    [10, 15],
    [15, 12],
    [15, 13],
    [44, 48],
    [24, 10],
    [30, 45],
    [85, 70],
    [71, 80],
    [60, 78],
    [55, 52],
    [80, 91]
])

# Configure model parameters
n_clusters = 3
fuzziness_parameter = 3

# Apply fuzzy c-means clustering
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X.T, n_clusters, fuzziness_parameter, error=0.005, maxiter=1000, init=None
)

# Predict cluster membership for each data point
cluster_membership = np.argmax(u, axis=0)

# Print the cluster centers
print('Cluster Centers:\n', cntr)

# Print the cluster membership for each data point
print('\nCluster Membership:', cluster_membership)

# Plot the clusters
plt.scatter(cntr[:, 0], cntr[:, 1], color=['red', 'green', 'blue'])
plt.scatter(X[:, 0], X[:, 1], c=cluster_membership)

plt.title('Fuzzy C-means. n_clusters=%s' % n_clusters)
plt.show()
