from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
pd.set_option('display.float_format', lambda x: '%.i' % x)

# Load the data
df = pd.read_csv('../assets/titanic_train.csv')

print(df.info())
print(df.describe())
print('# of passengers in original data:', df.shape[0])

# Analyze the data
sns.countplot(x='Survived', data=df, hue='Sex')
sns.countplot(x='Survived', data=df, hue='Pclass')
sns.countplot(x='SibSp', data=df)

df['Age'].plot.hist()
df['Fare'].plot.hist(bins=20)
plt.show()

# Wrangling (cleaning) the data

# Checking the null values
print(df.isnull().sum())
sns.heatmap(df.isnull(), yticklabels=False, cmap='viridis')

# Distribution of variables
sns.boxplot(x='Pclass', y='Age', data=df)

# Dropping columns and null values
df.drop(['Cabin'], axis=1, inplace=True)
df.dropna(inplace=True)
print(df.shape)

print(df.isnull().sum())
sns.heatmap(df.isnull(), yticklabels=False, cmap='viridis')
plt.show()


def encode_to_categorical_features():
    # Encoding text variables to categorical
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

    return df


def encode_to_binary_features():
    # Encoding and transposing text variables to binary variables
    sex = pd.get_dummies(df['Sex'], dtype='int', drop_first=True)
    embark = pd.get_dummies(df['Embarked'], dtype='int', drop_first=True)
    p_class = pd.get_dummies(df['Pclass'], dtype='int', drop_first=True)

    df_enc = pd.concat([df, sex, embark, p_class], axis=1)
    df_enc.drop(['PassengerId', 'Pclass', 'Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)

    return df_enc


df = encode_to_categorical_features()
# df = encode_to_binary_features()

# Converting columns' names to int
df.columns = df.columns.astype(str)

# Splitting the data
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Training the classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Predicting
y_pred = clf.predict(X_test)

# Evaluating
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
