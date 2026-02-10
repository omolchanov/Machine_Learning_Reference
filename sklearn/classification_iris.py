from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def classification_example():

    """
    1. Load dataset
    The Iris dataset is a classic multiclass classification dataset
    It contains 4 features (sepal length, sepal width, petal length, petal width)
    and 3 target classes (species of Iris flowers)
    """
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    """
    2. Split dataset into training and testing subsets
    We use 80% of data for training and 20% for testing
    The 'random_state' ensures reproducibility of the split
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    """
    3. Initialize and train the model
    RandomForestClassifier is an ensemble learning algorithm
    It builds multiple decision trees and combines their outputs
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    """
    4. Make predictions on the test set
    The model predicts the class labels for unseen samples
    """
    y_pred = model.predict(X_test)

    """
    5. Evaluate model performance
    classification_report shows precision, recall, f1-score, and accuracy
    for each class, giving a detailed picture of model performance
    """
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    classification_example()
