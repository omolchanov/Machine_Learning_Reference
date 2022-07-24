import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.tree._export import export_text
from sklearn.pipeline import Pipeline

# Configure Pandas
pd.set_option("display.precision", 2)
pd.set_option("expand_frame_repr", False)
# pd.set_option('display.max_rows', None)

df = pd.read_csv('../assets/churn-rate.csv')

# Prepare the dataframe
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

df['international plan'] = pd.factorize(df['international plan'])[0]
df['voice mail plan'] = pd.factorize(df['voice mail plan'])[0]
df['churn'] = df['churn'].astype("int")
y = df['churn']
df.drop(['state', 'churn', 'phone number', 'area code'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df.values, y, test_size=0.3, random_state=17)


def calculate_accuracy_score_tree(model):
    model.fit(X_train, y_train)
    y_pred = model_tree.predict(X_test)

    print('TREE CLASSIFIER: Test Split size: %s, Accuracy score: %s \n'
          % (len(X_test), accuracy_score(y_pred, y_test)))


def calculate_accuracy_score_knn(model):
    """
    PREPROCESSING: Features scaling for KNN classifier
    https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    https://coderlessons.com/tutorials/python-technologies/uznaite-mashinnoe-obuchenie-s-python/mashinnoe-obuchenie-podgotovka-dannykh
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print('KNN CLASSIFIER: Scaled Test Split size: %s, Accuracy score: %s \n'
          % (len(X_test_scaled), accuracy_score(y_pred, y_test)))


def identify_best_parameters_tree(model_tree, feature_names):
    """
    Identitfies the best parameters and best score for SKLearn tree classifier
    Calculates accuracy score
    Builds the resulting tree

    https://vc.ru/ml/147132-kak-avtomaticheski-podobrat-parametry-dlya-modeli-mashinnogo-obucheniya-ispolzuem-gridsearchcv
    https://stackoverflow.com/questions/44459845/gridsearchcv-best-score-meaning-when-scoring-set-to-accuracy-and-cv
    """

    tree_params = {'max_depth': range(1, 5), 'max_features': range(4, 15)}
    tree_grid = GridSearchCV(model_tree, tree_params, cv=5, n_jobs=1, verbose=True)
    tree_grid.fit(X_train, y_train)

    print('TREE CLASSIFIER RESULTS:\n Best Parameters: %s\n Best score: %s'
          % (tree_grid.best_params_, tree_grid.best_score_))

    print('Accuracy score: %s \n' % (accuracy_score(y_test, tree_grid.predict(X_test))))

    # Print the resulting tree
    tree_rules = export_text(tree_grid.best_estimator_, feature_names=feature_names)
    print(tree_rules)


def identify_best_parameters_random_forest(model):
    """
    Identitfies the best parameters and best score for SKLearn Random Forest classifier
    Calculates cross validation score
    Calculates accuracy score
    https://towardsdatascience.com/a-practical-guide-to-implementing-a-random-forest-classifier-in-python-979988d8a263
    """

    forest_params = {"max_depth": range(6, 12), "max_features": range(4, 19)}
    forest_grid = GridSearchCV(model, forest_params, cv=5, n_jobs=1, verbose=True)
    forest_grid.fit(X_train, y_train)

    print('RANDOM FOREST CLASSIFIER RESULTS:\n Best Parameters: %s\n Best score: %s'
          % (forest_grid.best_params_, forest_grid.best_score_))

    print('Accuracy score: %s' % (accuracy_score(y_test, forest_grid.predict(X_test))))

    print('Cross validation score: %s\n' % (cross_val_score(model, X_train, y_train, cv=5)))


def identify_best_parameters_knn(model):
    """
    Identitfies the best parameters and best score for SKLearn KNN classifier
    Calculates accuracy score
    """

    model_pipeline = Pipeline(
        [('scaler', StandardScaler()), ('knn', model)]
    )

    knn_params = {"knn__n_neighbors": range(1, 10)}
    knn_grid = GridSearchCV(model_pipeline, knn_params, cv=5, n_jobs=1, verbose=True)
    knn_grid.fit(X_train, y_train)

    print('KNN CLASSIFIER RESULTS:\n Best Parameters: %s\n Best score: %s'
          % (knn_grid.best_params_, knn_grid.best_score_))

    print('Accuracy score: %s \n' % (accuracy_score(y_test, knn_grid.predict(X_test))))


model_tree = DecisionTreeClassifier()
model_knn = KNeighborsClassifier()
model_forest = RandomForestClassifier(n_estimators=10, n_jobs=1, random_state=17)

calculate_accuracy_score_tree(model_tree)
calculate_accuracy_score_knn(model_knn)

identify_best_parameters_tree(model_tree, df.columns.values.tolist())
identify_best_parameters_knn(model_knn)
identify_best_parameters_random_forest(model_forest)
