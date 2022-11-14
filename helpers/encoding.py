from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

import pandas as pd
import matplotlib.pyplot as plt

# Configure Pandas
pd.set_option("display.precision", 2)
pd.set_option("expand_frame_repr", False)
pd.set_option('display.max_rows', None)

# Reading the dataset
df = pd.read_csv('../assets/bank_train.csv', encoding='utf-8')
labels = pd.read_csv('../assets/bank_train_target.csv')

# Displaying Values counts by Education
print(df["education"].value_counts())
df["education"].value_counts().plot.barh()
plt.show()

# Encoding Education categorical values to numeric features
label_encoder = LabelEncoder()
enc_education_values = label_encoder.fit_transform(df['education'])

pd.Series(enc_education_values).value_counts().plot.barh()
plt.title(dict(enumerate(label_encoder.classes_)))
plt.show()

# Encoding all categorical columns
categorical_columns = df.columns[df.dtypes == "object"]
for c in categorical_columns:
    df[c] = label_encoder.fit_transform(df[c])

print(df.head(10))

# One-hot encoding with encoded categorical columns
oh_encoder = OneHotEncoder(sparse=False)
oh_encoded_categorical_columns = pd.DataFrame(oh_encoder.fit_transform(df[categorical_columns]))

print(oh_encoded_categorical_columns.head(10))


# Logistic Regression with encoded values
def logistic_regression_result(features):

    model = LogisticRegression()

    X_train, X_test, y_train, y_test = train_test_split(features, labels)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=True))


logistic_regression_result(df[categorical_columns])
logistic_regression_result(oh_encoded_categorical_columns)
