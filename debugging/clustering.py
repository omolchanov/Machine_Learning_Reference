import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate synthetic 2D dataset
X, _ = make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,
    cluster_std=0.60,
    random_state=42
)

# Fit KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# Compute distances to assigned cluster center
# distances to all centers
all_distances = kmeans.transform(X)

# distance to an assigned center
assigned_distances = np.min(all_distances, axis=1)

# Print some stats
print(f"Min distance: {assigned_distances.min():.4f}")
print(f"Max distance: {assigned_distances.max():.4f}")
print(f"Mean distance: {assigned_distances.mean():.4f}")

# Plot histogram of distances
plt.figure(figsize=(8, 5))
plt.hist(assigned_distances, bins=30, color="skyblue", edgecolor="k")
plt.title("Distribution of Distances to Assigned Cluster Center")
plt.xlabel("Distance")
plt.ylabel("Number of points")
plt.show()

# Compute and print cluster sizes
unique, counts = np.unique(y_kmeans, return_counts=True)
print("\nCluster sizes:")
for cluster_id, count in zip(unique, counts):
    print(f"  Cluster {cluster_id}: {count} points")

plt.figure(figsize=(6, 4))
plt.bar(unique, counts, color="lightgreen", edgecolor="k")
plt.xlabel("Cluster ID")
plt.ylabel("Number of points")
plt.title("Cluster Sizes")
plt.show()

# Find the nearest point to each cluster center
print("\nNearest point to each cluster center:")
for i, center in enumerate(kmeans.cluster_centers_):
    distances_to_center = np.linalg.norm(X - center, axis=1)
    nearest_idx = np.argmin(distances_to_center)
    nearest_point = X[nearest_idx]
    print(f"  Cluster {i}: {nearest_point}")

# Scatter plot of clusters with nearest points highlighted
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=30, cmap="viridis", alpha=0.6, edgecolor="k")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c="red", s=200, marker="X", label="Centers")

# Highlight nearest points
for i, center in enumerate(kmeans.cluster_centers_):
    distances_to_center = np.linalg.norm(X - center, axis=1)
    nearest_idx = np.argmin(distances_to_center)
    plt.scatter(
        X[nearest_idx, 0], X[nearest_idx, 1],
        c="gold",
        s=150,
        marker="o",
        edgecolor="k",
        label=f"Nearest to center {i}" if i == 0 else ""
    )

plt.title("KMeans Clustering with Nearest Points to Centers")
plt.legend()
plt.show()

# Cluster boundaries (2D visualization)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=30, cmap="viridis", alpha=0.6, edgecolor="k")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c="red", s=200, marker="X", label="Centers")

# Highlight nearest points
for i, center in enumerate(kmeans.cluster_centers_):
    nearest_idx = np.argmin(np.linalg.norm(X - center, axis=1))
    plt.scatter(X[nearest_idx, 0], X[nearest_idx, 1], c="gold", s=150, marker="o", edgecolor="k",
                label=f"Nearest to center {i}" if i == 0 else "")

plt.title("KMeans Clustering with Cluster Boundaries")
plt.legend()
plt.show()
