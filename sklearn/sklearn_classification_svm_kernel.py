from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np


# Training dataset (features and labels)
X = np.array([
    [5, 5],
    [6, 5],
    [4, 4],
    [8, 9],
    [7, 6],
    [10, 7],
    [8, 1]
])

y = np.array([0, 0, 0, 1, 1, 1, 1])


# Non-linear classification using different SVM Kernel algorithms
def classify(model: SVC, x_pred):
    # Split the dataset (features and labels) onto trainings and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.50)

    # Train the model
    model.fit(x_train, y_train)

    # Making prediction with the test set
    prediction = model.predict(x_test)

    # Evaluating the model
    print('KERNEL TYPE: ', str(model.kernel).upper())
    print('Accuracy score: ', accuracy_score(prediction, y_test))
    print('Confusion matrix: \n', confusion_matrix(prediction, y_test))
    print('Classification report: \n', classification_report(prediction, y_test))

    # Making a prediction
    y_pred = model.predict(x_pred)
    print('Prediction \n Features: %s \n Prediction Result: %s' % (str(x_pred).replace('\n', ''), y_pred))
    print('====================================================')

    return y_pred


# Visualizng the dataset and the predictions
def visualize(model, x_pred, y_pred):
    plt.title('SVM Classification. KERNEL TYPE: ' + str(model.kernel).upper())
    plt.xlabel("X")
    plt.ylabel("Y")

    x_data = np.concatenate((X, x_pred))
    y_data = np.concatenate((y, y_pred))

    # Label '0' is marked with black color
    # Label '1' is marked with yellow color
    plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)

    for i, txt in enumerate(x_data):
        plt.annotate(str(x_data[i]), (x_data[i]))

    h = 0.2 # step size
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.show()


x_pred = np.array([
    [0, 0],
    [10, 20],
    [3.5, 7.2],
    [6.9, 8],
    [1, 2],
    [10, 2],
    [-1, -5]
])

# Classification with Polynomial Kernel
model = SVC(kernel='poly', degree=8)
y_pred = classify(model, x_pred)
visualize(model, x_pred, y_pred)

# Classification with Gaussian Kernel
model = SVC(kernel='rbf')
classify(model, x_pred)
visualize(model, x_pred, y_pred)

# Classification with Sigmoid Kernel
model = SVC(kernel='sigmoid')
classify(model, x_pred)
visualize(model, x_pred, y_pred)
