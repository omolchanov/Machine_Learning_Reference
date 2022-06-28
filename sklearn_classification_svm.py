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
    [10, 7]
])

y = np.array([0, 0, 0, 1, 1, 1])

# Classify the dataset with Support Vector Machines algorithm
model = SVC(kernel='linear')

# Split the dataset (features and labels) onto trainings and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Train the model
model.fit(x_train, y_train)

# Making prediction with the test set
prediction = model.predict(x_test)

# Evaluating the model
print('SVC Accuracy score: ', accuracy_score(prediction, y_test))
print('SVC Confusion matrix: \n', confusion_matrix(prediction, y_test))
# print('SVC Classification report: \n', classification_report(prediction, y_test))

# Making a prediction
x_pred = np.array([
    [0, 0],
    [10, 20],
    [3.5, 7.2],
    [6.9, 8],
    [1, 2],
    [10, 2],
    [-1, -5]
])
y_pred = model.predict(x_pred)
print('SVC Prediction - Features: %s | Prediction Result: %s' % (str(x_pred).replace('\n', ''), y_pred))


# Visualizng the dataset and the predictions
plt.title('Classification with SVC algorithm')
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

# Classification boundary visualization
# https://medium.com/geekculture/svm-classification-with-sklearn-svm-svc-how-to-plot-a-decision-boundary-with-margins-in-2d-space-7232cb3962c0
w = model.coef_[0]
b = model.intercept_[0]
x_points = np.linspace(min(x_data[:, 0]), max(x_data[:, 0]), 200)
y_points = -(w[0] / w[1]) * x_points - b / w[1]
plt.plot(x_points, y_points, c='r')

plt.show()
