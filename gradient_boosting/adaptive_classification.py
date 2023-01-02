from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets

import matplotlib.pyplot as plt

# Prepare the dataset
dataset = datasets.load_breast_cancer()
X = dataset.data
y = dataset.target

# Scaling the data and reducing dimensionality
std_scaler = StandardScaler()
X_scaled = std_scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plotting initial dataset
plt.figure(figsize=(15, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)

plt.title('Initial dataset')
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3)


def adaptive_boosting_classify(estimator_model=None):

    # Fitting the model
    model = AdaBoostClassifier(n_estimators=5, base_estimator=estimator_model)
    model.fit(X_train, y_train)

    # Plotting predictions/results for each estimator
    for i, estimator in enumerate(model.estimators_):
        y_pred = estimator.predict(X_test)

        plt.figure(figsize=(15, 7))
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
        plt.title('Estimator #%s. Error: %s' % (i, round(model.estimator_errors_[i], 3)))
        plt.show()

    # Plotting final prediction
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred).__round__(3)

    plt.figure(figsize=(15, 7))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
    plt.title('Final Prediction. Accuracy: %s' % score)
    plt.show()


# Launching gradient boosting with different estimator models
cl_svc = SVC(probability=True, kernel='linear')
cl_tree = DecisionTreeClassifier(max_depth=5, max_features=5)

adaptive_boosting_classify(estimator_model=cl_svc)
