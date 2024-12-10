from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import pandas as pd

df = pd.read_csv('../assets/suv_data.csv')

# Defining independent and dependent variables
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Scaling the data
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

# Training the classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Predicting
y_pred = clf.predict(X_test)

# Evaluating
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
