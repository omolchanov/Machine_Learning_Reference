import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# https://towardsdatascience.com/understanding-random-forest-58381e0602d2

# Read and prepare the dataset
df = pd.read_csv('../assets/churn-rate.csv')

columns_to_float = [
    'total day minutes',
    'total day charge',
    'total eve minutes',
    'total eve charge',
    'total night minutes',
    'total night charge',
    'total intl minutes',
    'total intl charge'
]

for i, name in enumerate(columns_to_float):
    df[name] = df[name].str.replace(",", ".").astype('float64')

df['churn'] = df['churn'].map({False: 0, True: 1})

X = df.drop(['state', 'area code', 'phone number', 'international plan', 'voice mail plan', 'churn'], axis=1)
y = df['churn'].values


def evaluate_single_cv_model():

    # Initialize a stratified split of our dataset for the validation process
    # https://www.geeksforgeeks.org/stratified-k-fold-cross-validation/
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    model = RandomForestClassifier(random_state=0, n_jobs=1)

    cv_score = cross_val_score(model, X, y, cv=skf)
    print('CV accuracy score: ', np.mean(cv_score).round(2))


def evaluate_hyperparameter_n_estimators():
    """
    Performs CV and finds the best number of n_estimators of RandomForest Classified basing on the max CV score
    n_estimators — the number of trees in the forest
    """

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    n_estimators = [5, 10, 15, 20, 30, 50, 75, 100]
    cv_scores = []

    for i in n_estimators:
        model = RandomForestClassifier(n_estimators=i, random_state=0, n_jobs=-1)
        cv_score = cross_val_score(model, X, y, cv=skf)

        cv_scores.append(np.mean(cv_score))

    print('Best CV score is %s with n_estimators = %s ' %
          (np.max(cv_scores).round(2), n_estimators[cv_scores.index(np.max(cv_scores))]))

    plt.style.use('ggplot')
    plt.plot(n_estimators, cv_scores, label='CV score')

    plt.xlabel('Number of estimators')
    plt.ylabel('Accuracy')

    plt.legend()
    plt.show()


def evaluate_hyperparameter_max_depth():
    """
    Performs CV and finds the best number of max_depth of RandomForest Classified basing on the max CV score
    max_depth — the maximum depth of the tree
    """

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    max_depths = [3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 24]
    cv_scores = []

    for i in max_depths:
        model = RandomForestClassifier(n_estimators=50, max_depth=i, random_state=0, n_jobs=-1)
        cv_score = cross_val_score(model, X, y, cv=skf)

        cv_scores.append(np.mean(cv_score))

    print('Best CV score is %s with max_depth = %s ' %
          (np.max(cv_scores).round(2), max_depths[cv_scores.index(np.max(cv_scores))]))

    plt.style.use('ggplot')
    plt.plot(max_depths, cv_scores, label='CV score')

    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')

    plt.legend()
    plt.show()


def evaluate_hyperparameter_min_samples_leaf():
    """
    Performs CV and finds the best number of min_samples_leaf of RandomForest Classifier basing on the max CV score
    min_samples_leaf — the minimum number of samples required to be at a leaf node
    """

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    min_samples_leaves = [1, 3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 24]
    cv_scores = []

    for i in min_samples_leaves:
        model = RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_leaf=i, random_state=0, n_jobs=-1)
        cv_score = cross_val_score(model, X, y, cv=skf)

        cv_scores.append(np.mean(cv_score))

    print('Best CV score is %s with min_samples_leaf = %s ' %
          (np.max(cv_scores).round(2), min_samples_leaves[cv_scores.index(np.max(cv_scores))]))

    plt.style.use('ggplot')
    plt.plot(min_samples_leaves, cv_scores, label='CV score')

    plt.xlabel('Min Samples Leaf')
    plt.ylabel('Accuracy')

    plt.legend()
    plt.show()


def evaluate_hyperparameter_max_features():
    """
    Performs CV and finds the best number of max_features of RandomForest Classifier basing on the max CV score
    max_features — the number of features to consider when looking for the best split
    """

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    max_features = [2, 4, 6, 8, 10, 12, 14, 16]
    cv_scores = []

    for i in max_features:
        model = RandomForestClassifier(
            n_estimators=50, max_depth=20, min_samples_leaf=1, max_features=i, random_state=0, n_jobs=-1
        )

        cv_score = cross_val_score(model, X, y, cv=skf)

        cv_scores.append(np.mean(cv_score))

    print('Best CV score is %s with max_features = %s ' %
          (np.max(cv_scores).round(2), max_features[cv_scores.index(np.max(cv_scores))]))

    plt.style.use('ggplot')
    plt.plot(max_features, cv_scores, label='CV score')

    plt.xlabel('Max Features')
    plt.ylabel('Accuracy')

    plt.legend()
    plt.show()


def evaluate_hyperparameters_with_cv_grid_search():
    parameters = {
        'n_estimators': [5, 10, 15, 20, 30, 50, 75, 100],
        'max_depth': [3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 24],
        'min_samples_leaf': [1, 3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 24],
        'max_features': [2, 4, 6, 8, 10, 12, 14, 16]
    }

    model = RandomForestClassifier(random_state=0, n_jobs=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    gcv = GridSearchCV(model, parameters, n_jobs=-1, cv=skf, verbose=1)
    gcv.fit(X, y)

    print(gcv.best_params_, gcv.best_score_)


evaluate_single_cv_model()
evaluate_hyperparameter_n_estimators()
evaluate_hyperparameter_max_depth()
evaluate_hyperparameter_min_samples_leaf()
evaluate_hyperparameter_max_features()
evaluate_hyperparameters_with_cv_grid_search()
