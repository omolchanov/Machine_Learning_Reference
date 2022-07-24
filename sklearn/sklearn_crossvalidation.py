import warnings
warnings.filterwarnings("ignore")

import pprint
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import numpy as np

# Configure Pandas
pd.set_option("display.precision", 2)
pd.set_option("expand_frame_repr", False)
pd.set_option('display.max_rows', None)


df = pd.read_csv('../assets/breast_dataset.csv')

# Separate features and target variable
X = df.iloc[:, 2:-1].values

# The number 0 represents benign, while 1 represents malignant
y = df['diagnosis'].map({'B': 0, 'M': 1})


def cross_validation(X, y, model, cv=5):
    """
    Function to perform 5 Folds Cross-Validation
    :param cv: Determines the number of folds for cross-validation
    :return result

    fit_time - The time for fitting the estimator on the train set for each cv split
    score_time - The time for scoring the estimator on the test set for each cv split.

    F1 Score, Prescsion, Recall metrics
    https://towardsdatascience.com/essential-things-you-need-to-know-about-f1-score-dbd973bf1a3
    """

    scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                             X=X,
                             y=y,
                             cv=cv,
                             scoring=scoring,
                             return_train_score=True)

    return results


def visualize(results):
    x_labels = [1, 2, 3, 4, 5]
    width = 0.25

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    data = pd.DataFrame(results)

    # F1-score is the harmonic mean of precision and recall. The higher the precision and recall, the higher the
    # F1-score. F1-score ranges between 0 and 1.
    # The closer it is to 1, the better the model.
    axs[0][0].bar(np.arange(len(x_labels)), data['train_f1'], width, label="Train F1")
    axs[0][0].bar(np.arange(len(x_labels)) + width, data['test_f1'], width, label="Test F1")
    axs[0][0].title.set_text('F1 Score')
    axs[0][0].legend()

    # https://pythonguides.com/scikit-learn-accuracy-score/
    axs[0][1].bar(np.arange(len(x_labels)), data['train_accuracy'], width, label="Train Accuracy")
    axs[0][1].bar(np.arange(len(x_labels)) + width, data['test_accuracy'], width, label="Test Accuracy")
    axs[0][1].title.set_text('Accuracy')
    axs[0][1].legend()

    # Precision measures the extent of error caused by False Positives (FPs)
    axs[1][0].bar(np.arange(len(x_labels)), data['train_precision'], width, label="Train Precision")
    axs[1][0].bar(np.arange(len(x_labels)) + width, data['test_precision'], width, label="Test Precision")
    axs[1][0].title.set_text('Precision')
    axs[1][0].legend()

    # Recall measures the extent of error caused by False Negatives (FNs)
    axs[1][1].bar(np.arange(len(x_labels)), data['train_recall'], width, label="Train Recall")
    axs[1][1].bar(np.arange(len(x_labels)) + width, data['test_recall'], width, label="Test Recall")
    axs[1][1].title.set_text('Recall')
    axs[1][1].legend()

    plt.show()


"""
https://stats.stackexchange.com/questions/52274/how-to-choose-a-predictive-model-after-k-fold-cross-validation

When you do K-fold cross validation, you are testing how well your model is able to get trained by some data and then 
predict data it hasn't seen. We use cross validation for this because if you train using all the data you have, you have 
none left for testing. You could do this once, say by using 80% of the data to train and 20% to test, but what if the 
20% you happened to pick to test happens to contain a bunch of points that are particularly easy (or particularly hard) 
to predict? We will not have come up with the best estimate possible of the models ability to learn and predict

The purpose of cross-validation is not to come up with our final model. We don't use these K instances of our trained 
model to do any real prediction. For that we want to use all the data we have to come up with the best model possible. 
The purpose of cross-validation is model checking, not model building.
"""

model = KNeighborsClassifier(n_neighbors=3)
result = cross_validation(X, y, model)

pp = pprint.PrettyPrinter(depth=1)
pp.pprint(result)

visualize(result)
