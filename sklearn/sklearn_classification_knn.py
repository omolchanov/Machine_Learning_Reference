from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Training dataset (features and labels)
X = np.array([
    [5, 5],
    [6, 5],
    [4, 4],
    [8, 9],
    [7, 6],
    [10, 7]
])

y = np.array([0, 0, 0, 1, 1, 1])


# Classify the dataset with K-neighboors algorithm
def classify_KNN(features, labels, x_pred):
    model = KNeighborsClassifier(n_neighbors=1)

    # Split the dataset (features, labels) onto training and test sets
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=27)

    # Training the model with the training set (X_train, y_train)
    model.fit(x_train, y_train)

    # Making prediction with the test set
    prediction = model.predict(x_test)

    # Evaluating the model
    print('KNN Accuracy score: %s', accuracy_score(prediction, y_test))
    print('KNN Confusion matrix: \n', confusion_matrix(prediction, y_test))

    # Making a prediction
    y_pred = model.predict(x_pred)
    print('KNN Prediction - Features: %s | Prediction Result: %s' % (str(x_pred).replace('\n', ''), y_pred))

    return y_pred


x_pred = np.array([
    [0, 0],
    [10, 20],
    [3.5, 7.2],
    [6.9, 8]
])
y_pred = classify_KNN(X, y, x_pred)


# Visualizng the dataset and the predictions
plt.title('Classification with KNN algorithm')
plt.xlabel("X")
plt.ylabel("Y")

x_data = np.concatenate((X, x_pred))
y_data = np.concatenate((y, y_pred))

# Dataset visualization
# Label '0' is marked with black color
# Label '1' is marked with yellow color
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)

for i, txt in enumerate(x_data):
    plt.annotate(str(y_data[i]) + ' ' + str(x_data[i]), (x_data[i]))

plt.show()
