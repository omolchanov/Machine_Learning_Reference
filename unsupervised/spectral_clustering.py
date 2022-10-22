from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

import pandas as pd

# Configure Pandas
pd.set_option("display.precision", 2)
pd.set_option("expand_frame_repr", False)
pd.set_option('display.max_rows', None)

# Manual: https://www.geeksforgeeks.org/ml-spectral-clustering/

# Preparing the dataset
df = pd.read_csv('../assets/cc_dataset.csv')

X = df.drop('CUST_ID', axis=1)
X.fillna(method='ffill', inplace=True)


# Preprocessing the data to make it visualizable
# Scaling the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Normalizing the Data
X_normalized = normalize(X_scaled)

# Reducing the dimensions of the data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)

# Building the clustering model

# Affinity == rbf
model = SpectralClustering(n_clusters=2, affinity='rbf')
labels = model.fit_predict(X_pca)

print('Silhouette Score | rfb: ', silhouette_score(X, labels).__round__(3))

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
plt.title('Affinity == rfb')
plt.show()


# Affinity == nearest_neighbors
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors')
labels = model.fit_predict(X_pca)

print('Silhouette Score | nearest_neighbors: ', silhouette_score(X, labels).__round__(3))

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
plt.title('Affinity == nearest_neighbors')
plt.show()
