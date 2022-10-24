from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
import numpy as np

# Manual: https://www.geeksforgeeks.org/ml-hierarchical-clustering-agglomerative-and-divisive-clustering/

# Dataset with coordinates of stores
X = np.array([[5, 3], [10, 15], [15, 12], [24, 10], [30, 45], [85, 70], [71, 80], [60, 78], [55, 52], [80, 91]])

# Build clusters
model = AgglomerativeClustering(affinity='euclidean', linkage='ward')
y = model.fit(X)

y_pred = model.fit_predict(X)
print('Silhouette Score: ', silhouette_score(X, y_pred), '\n')


# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=y.labels_)

plt.title(model.get_params())
plt.show()

# Plot dendogram
distance_mat = pdist(X)
Z = hierarchy.linkage(distance_mat, 'ward')

plt.figure()
hierarchy.dendrogram(Z, color_threshold=0.9)

plt.title('Cluster Dendogram')
plt.show()
