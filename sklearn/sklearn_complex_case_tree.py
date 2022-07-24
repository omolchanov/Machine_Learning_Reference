from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree._export import export_text
import matplotlib.pyplot as plt
import numpy as np


# Generates random linearly separated data
def generate_data(n, x_min, x_max):
    X, y = [], []

    for i in range(n):
        x1 = np.random.randint(x_min, x_max)
        x2 = np.random.randint(x_min, x_max)

        if np.abs(x1 - x2) > 0.5:
            X.append([x1, x2])
            y.append(np.sign(x1 - x2))

    return np.array(X), np.array(y)


# Third-party function for building pyplot colormesh
def get_grid(data):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))


X, y = generate_data(10, 0, 50)


def tree_classify(model):
    model.fit(X, y)

    xx, yy = get_grid(X)
    y_pred = model_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    tree_rules = export_text(model_tree)
    print(tree_rules)

    plt.pcolormesh(xx, yy, y_pred, cmap="autumn")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn', edgecolors="black")
    plt.title('Easy task. Too long for Descision Tree Classifier')

    plt.show()


def knn_classify(model):
    model.fit(X, y)

    xx, yy = get_grid(X)
    y_pred = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.pcolormesh(xx, yy, y_pred, cmap="autumn")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn', edgecolors="black")
    plt.title('Easy task. Very very long for KNN Classifier')

    plt.show()


model_tree = DecisionTreeClassifier()
model_knn = KNeighborsClassifier(n_neighbors=1)

tree_classify(model_tree)
knn_classify(model_knn)
