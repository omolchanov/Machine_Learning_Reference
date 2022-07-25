from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np


def generate_data(n_obj, n_feat):
    np.random.seed(17)

    y = np.random.choice([-1, 1], size=n_obj)

    x1 = 0.3 * y
    x_other = np.random.random(size=[n_obj, n_feat - 1])

    return np.hstack([x1.reshape([n_obj, 1]), x_other]), y


X, y = generate_data(1000, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

cv_scores, accuracy_scores = [], []
n_neighb = [1, 2, 3, 5] + list(range(50, 551, 50))

for k in n_neighb:
    knn_pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k))])
    cv_scores.append(np.mean(cross_val_score(knn_pipe, X_train, y_train, cv=5)))

    knn_pipe.fit(X_train, y_train)
    accuracy_scores.append(accuracy_score(y_test, knn_pipe.predict(X_test)))

print('Cross validation scores: %s\nAccuracy scores: %s' % (cv_scores, accuracy_scores))

# Visualizng results
plt.plot(n_neighb, cv_scores, label="CV")
plt.plot(n_neighb, accuracy_scores, label="holdout")

plt.title("Easy task. kNN fails")
plt.legend()
plt.show()

tree = DecisionTreeClassifier(random_state=17, max_depth=1)
tree_cv_score = np.mean(cross_val_score(tree, X_train, y_train, cv=5))
tree.fit(X_train, y_train)
tree_holdout_score = accuracy_score(y_test, tree.predict(X_test))
print("Decision tree. CV: {}, holdout: {}".format(tree_cv_score, tree_holdout_score))
