import warnings
warnings.filterwarnings("ignore")

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import numpy as np


def pca_iris_dataset():

    # Prepare the dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Initial prediction
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = DecisionTreeClassifier(max_depth=2, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('Initial Accuracy score %s' % (accuracy_score(y_test, y_pred).__round__(3)))

    # Prediction with PCA
    pca = PCA(n_components=2)

    std_scaler = StandardScaler()
    X_scaled = std_scaler.fit_transform(X)

    X_pca = pca.fit_transform(X_scaled)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('PCA Accuracy score %s' % (accuracy_score(y_test, y_pred).__round__(3)))
    print(pca.explained_variance_ratio_)

    # Plotting data
    plt.plot(X_pca[y == 0, 0], X_pca[y == 0, 1], "bo", label="Setosa")
    plt.plot(X_pca[y == 1, 0], X_pca[y == 1, 1], "go", label="Versicolour")
    plt.plot(X_pca[y == 2, 0], X_pca[y == 2, 1], "ro", label="Virginica")
    plt.legend(loc=0)

    plt.show()


def pca_digits_dataset():

    # Prepare the dataset
    data = datasets.load_digits()
    X, y = data.data, data.target

    # Reducing 64-dimensional data to 2D shape
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(12, 10))
    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y,
        alpha=0.7,
        s=40,
        edgecolor='none',
        cmap=plt.cm.get_cmap("nipy_spectral", 10),
    )
    plt.colorbar()
    plt.title('PCA - Projecting 64-dimensional data to 2D')
    plt.show()

    # Plotting total explained variance
    plt.figure(figsize=(12, 10))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), color="k", lw=2)

    plt.xlabel("Number of components")
    plt.ylabel("Total explained variance")
    plt.title('Plotting total explained variance')
    plt.show()


def tsne_digits_datasets():

    # Prepare the dataset
    data = datasets.load_digits()
    X, y = data.data, data.target

    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(12, 10))
    plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=y,
        alpha=0.7,
        s=40,
        edgecolor='none',
        cmap=plt.cm.get_cmap("nipy_spectral", 20)
    )
    plt.colorbar()
    plt.title('TSNE - Projecting 64-dimensional data to 2D')
    plt.show()


pca_digits_dataset()
tsne_digits_datasets()
