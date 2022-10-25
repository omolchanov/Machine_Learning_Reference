from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)

from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
import numpy as np

# Manual: https://www.geeksforgeeks.org/ml-hierarchical-clustering-agglomerative-and-divisive-clustering/

# Dataset with coordinates of stores
X = np.array([[5, 3], [10, 15], [15, 12], [24, 10], [30, 45], [85, 70], [71, 80], [60, 78], [55, 52], [80, 91]])
y_true = [0, 1, 0, 0, 1, 1, 1, 1, 1, 1]

# Build clusters
model = AgglomerativeClustering(affinity='euclidean', linkage='ward')
y = model.fit(X)

# Calculatng accuracy metrics
# Manual: https://mlcourse.ai/book/topic07/topic7_pca_clustering.html#accuracy-metrics

y_pred = model.fit_predict(X)

#  The silhouette distance shows to which extent the distance between the objects of the same class differ from the
#  mean distance between the objects from different clusters. This coefficient takes values in the [-1, 1] range.
#  Values close to -1 correspond to bad clustering results while values closer to 1 correspond to well-defined clusters
print('Silhouette Score: ', silhouette_score(X, y_pred), '\n')

# ARI takes on values in the [-1, 1] range. Negative values indicate the independence of splits, and positive values
# indicate that these splits are consistent (they match ARI=1)
print('Adjusted Rand Index (ARI): ', adjusted_rand_score(y_true, y_pred))

# The AMI lies in the [0,1] range. Values close to zero mean the splits are independent, and those close to 1 mean they
# are similar (with complete match at AMI=1)
print('Adjusted mutual information (AMI): ', adjusted_mutual_info_score(y_true, y_pred))

# Homogeneity and Completeness. Both lie in the [0,1] range, and values closer to 1 indicate more accurate
# clustering results
print('Homogenity score: ', homogeneity_score(y_true, y_pred))
print('Completness score: ', completeness_score(y_true, y_pred))

# V-measure is a combination of Homogeneity and Completeness, and is their harmonic mean
print('V-measure: ', v_measure_score(y_true, y_pred))

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
