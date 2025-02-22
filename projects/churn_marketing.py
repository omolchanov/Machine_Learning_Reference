# https://www.youtube.com/watch?v=ajRfPY7s3CE
# https://www.evidentlyai.com/classification-metrics/accuracy-precision-recall
# https://plotly.com/graphing-libraries/

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np

import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import plot

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)

# PnL Configuration
LOSES = -800
RETENTION_COSTS = 200
PROFIT = 400

MARKETING_BUDGET = 5000
COST_PER_CLIENT = 30

df = pd.read_csv('../assets/churn-rate.csv', decimal=",")


# Data visualization
def visualize_data():
    figures = [
        px.bar(
            df.groupby(['churn']).
            agg({'churn': 'size'})['churn'].
            sort_values(ascending=False),
            color=['blue', 'red']),

        px.bar(
            df[df['churn'] == True].
            groupby(['state']).agg({'churn': 'size'})['churn'].
            sort_values(ascending=False),
        ),

        px.bar(
            df[df['churn'] == True].
            groupby(['customer service calls']).
            agg({'customer service calls': 'size'})['customer service calls'].
            sort_values(ascending=False),
            orientation='h'),
    ]

    figures_names = [
        'Churned vs non-churned customers',
        'Churned customers by state',
        'Customer services calls by state'
    ]

    fig = make_subplots(rows=len(figures), cols=1, subplot_titles=figures_names)

    for i, figure in enumerate(figures):
        for trace in range(len(figure['data'])):
            fig.add_trace(figure['data'][trace], row=i + 1, col=1)

    fig.update_layout(showlegend=False)

    plot(fig)


# Data preparation
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X = X.drop(['area code', 'phone number'], axis=1)

X['state'] = LabelEncoder().fit_transform(X['state'])
X['international plan'] = LabelEncoder().fit_transform(X['international plan'])
X['voice mail plan'] = LabelEncoder().fit_transform(X['voice mail plan'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)


def get_clients_sample_size(criteria: int | str = 10):
    if type(criteria) is int:
        return criteria

    if criteria == 'marketing_budget':
        return int(MARKETING_BUDGET / COST_PER_CLIENT)


def get_clf(clf, evaluate=True):
    print(clf)
    clf.fit(X_train, y_train)

    if evaluate is True:
        """
        Recall is a metric that measures how often a machine learning model correctly identifies positive instances 
        (true positives, TP) from all the actual positive samples in the dataset.
        Recall = TP / (TP + FN)
        
        Precision is a metric that measures how often a machine learning model correctly predicts the positive class. 
        You can calculate precision by dividing the number of correct positive predictions (true positives) by the 
        total number of instances the model predicted as positive (both true and false positives).
        Precision = TP / (TP + FP)
        """
        y_pred = clf.predict(X_test)
        print(classification_report(y_pred, y_test))

    return clf


def predict_pnl(clf, X_pred):
    get_clf(clf, evaluate=True)

    y_pred_proba = clf.predict_proba(X_pred)[:, 1]
    y_pred = clf.predict(X_pred)

    df_results = pd.DataFrame({
        'customer_id': X_pred.index,
        'churn': y_pred,
        'churn probability': y_pred_proba,
    })

    df_results['PnL'] = np.where(df_results["churn"] == 1, LOSES, PROFIT - RETENTION_COSTS)
    pnl = df_results['PnL'].sum()

    print(df_results)
    print('Sample size:', df_results.shape[0])
    print('PnL:', pnl)


if __name__ == '__main__':
    visualize_data()
    n_sample = get_clients_sample_size()
    X_pred = X.sample(n=n_sample)

    predict_pnl(DecisionTreeClassifier(), X_pred)
    predict_pnl(LogisticRegression(max_iter=10000), X_pred)
