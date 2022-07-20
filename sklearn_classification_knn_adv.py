import warnings
warnings.filterwarnings("ignore")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Training dataset (features and labels)
X = np.array([
    [5, 5],
    [6, 5],
    [4, 4],
    [8, 9],
    [7, 6],
    [10, 7],
    [3, 3]
])

y = np.array([0, 0, 0, 1, 1, 1, 0])


# Classify the dataset with K-neighboors algorithm
def classify_KNN(features, labels, x_pred, n_neighbors, weights, algorithm, metric):
    """
    @weights: 'uniform' (all weights are equal), 'distance' (the weight is inversely proportional to the distance from
    the test sample), or any other user-defined function

    @algorithm: 'brute', 'ball_tree' , 'KD_tree', or 'auto'. In the first case, the nearest neighbors for each
    test case are computed by a grid search over the training set. In the second and third cases, the distances between
    the examples are stored in a tree to accelerate finding nearest neighbors. If you set this parameter to auto,
    the right way to find the neighbors will be automatically chosen based on the training set.

    Brute force algorithm is better suited for small datasets while ball tree and KD tree for larger
    Brute force algorithm: https://medium.com/@m.yuvarajmp/exploring-knn-algorithm-brute-force-783656adef57
    Ball tree and KD tree algorithms:
    https://towardsdatascience.com/tree-algorithms-explained-ball-tree-algorithm-vs-kd-tree-vs-brute-force-9746debcd940

    @metric: minkowski, manhattan , euclidean, chebyshev, or other.
    https://machinelearningmastery.com/distance-measures-for-machine-learning/
    """

    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, metric=metric)

    # Split the dataset (features, labels) onto training and test sets
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=27)

    # Training the model with the training set (X_train, y_train)
    model.fit(x_train, y_train)

    # Making prediction with the test set
    prediction = model.predict(x_test)

    # Evaluating the model
    print('MODEL PARAMETERS: \n'
          'Number of Neighbors: %s \n'
          'Weights: %s \n'
          'Algorithm: %s \n'
          'Metric: %s \n'
          % (n_neighbors, weights, algorithm, metric))

    print('Accuracy score: ', accuracy_score(prediction, y_test))
    print('Confusion matrix: \n', confusion_matrix(prediction, y_test))
    print('Classification report: \n', classification_report(prediction, y_test))

    # Making a prediction
    y_pred = model.predict(x_pred)
    print('KNN Prediction - Features: %s | Prediction Result: %s \n' % (str(x_pred).replace('\n', ''), y_pred))

    return y_pred, model


# Visualizng the dataset and the predictions
def visualize(X, y, x_pred, y_pred, model):

    plt.title('MODEL PARAMETERS: '
              'Number of Neighbors=%s; '
              'Weights=%s; '
              'Algorithm=%s; '
              'Metric=%s; '
              % (model.n_neighbors, model.weights, model.algorithm, model.metric))

    plt.xlabel("X")
    plt.ylabel("Y")

    x_data = np.concatenate((X, x_pred))
    y_data = np.concatenate((y, y_pred))

    sns.scatterplot(x_data[:, 0], x_data[:, 1], hue=y_data)
    sns.regplot(x_pred[:, 0], x_pred[:, 1], scatter=False)

    for i, txt in enumerate(x_data):
        plt.annotate(str(y_data[i]) + ' ' + str(x_data[i]), (x_data[i]))

    plt.show()


x_pred = np.array([
    [0, 0],
    [10, 20],
    [3.5, 7.2],
    [6.9, 8]
])

y_pred1, model = classify_KNN(X, y, x_pred, n_neighbors=2, weights='uniform', algorithm='brute', metric='minkowski')
visualize(X, y, x_pred, y_pred1, model)

y_pred2, model = classify_KNN(X, y, x_pred, n_neighbors=2, weights='uniform', algorithm='ball_tree', metric='manhattan')
visualize(X, y, x_pred, y_pred2, model)
