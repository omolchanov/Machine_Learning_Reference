from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np


# Print lists without truncating
np.set_printoptions(threshold=np.inf)

# Load digits dataset
data = load_digits()

# Pictures here are 8x8 matrices (intensity of white color for each pixel).
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)


def visualize_digits(X):
    f, axes = plt.subplots(1, 10, sharey=True, figsize=(10, 10))
    for i in range(10):
        # matrix is 'unfolded' into a vector of length 64, and we obtain a feature description of an object
        # https://pythobyte.com/matplotlib-imshow-76005/
        axes[i].imshow(X[i, :].reshape([8, 8]), cmap='Greys')

    plt.show()


def evaluate_model():
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('Original Model accuracy: ', accuracy_score(y_test, y_pred))

    # Tuning the model
    model_pipe = Pipeline(
        [('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))]
    )
    model_pipe.fit(X_train, y_train)

    y_pred = model_pipe.predict(X_test)
    print('Tuned Model accuracy: ', accuracy_score(y_test, y_pred))

    # Cross validation of the model
    cv_score = cross_val_score(KNeighborsClassifier(n_neighbors=1), X_train, y_train, cv=5)
    print('Cross Validated Model ', np.mean(cv_score))


# Predicting digits
model = KNeighborsClassifier()
model.fit(X_train, y_train)

x_pred = [[0., 0., 2., 13., 8., 0., 0., 0., 0., 0., 6., 16., 16., 6., 0., 0., 0., 0.,
          5., 15., 13., 11., 0., 0., 0., 0., 0., 7., 16., 15., 0., 0., 0., 0., 0., 0.,
          0., 14., 3., 0., 0., 0., 0., 0., 0., 7., 11., 0., 0., 0., 0., 3., 4., 4.,
          16., 2., 0., 0., 2., 15., 13., 14., 13., 2.]]

y_pred = model.predict(np.array(x_pred))
print('Matrix: %s\nPredicted digit: %s' % (x_pred, y_pred[0]))

plt.imshow(np.array(x_pred).reshape([8, 8]), cmap='BuGn')
plt.show()
