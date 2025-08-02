"""
Customer Churn Prediction
Dataset: User accounts and activity
Goal: Predict whether a customer will leave (binary classification)
Tools: Decision trees or logistic regression
"""

import warnings
warnings.filterwarnings("ignore")

import duckdb
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix


if __name__ == '__main__':

    # Generate synthetic user data
    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        'user_id': range(1, n + 1),
        'tenure_months': np.random.randint(1, 48, n),
        'num_logins': np.random.poisson(10, n),
        'support_tickets': np.random.poisson(1.5, n),
        'premium_plan': np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
    })

    # Simulate churn probability based on features
    prob_churn = (
        0.3 * (df['tenure_months'] < 6) +
        0.3 * (df['support_tickets'] > 3) +
        0.2 * (df['num_logins'] < 5) +
        0.1 * (1 - df['premium_plan']) +
        np.random.normal(0, 0.05, n)
    )

    df['churn'] = (prob_churn > 0.5).astype(int)

    # Step 2: Store in DuckDB
    con = duckdb.connect()
    con.register("churn", df)
    con.execute("CREATE TABLE users AS SELECT * FROM churn")

    # Feature engineering with complex SQL
    query = """
    SELECT
        tenure_months,
        num_logins,
        support_tickets,
        premium_plan,
    
        CASE 
            WHEN tenure_months < 6 THEN 'new'
            WHEN tenure_months BETWEEN 6 AND 24 THEN 'mid'
            ELSE 'loyal'
        END AS tenure_bucket,
    
        num_logins / NULLIF(tenure_months, 0) AS logins_per_month,
        support_tickets / NULLIF(tenure_months, 0) AS tickets_per_month,
    
        CASE WHEN num_logins > 15 THEN 1 ELSE 0 END AS is_high_login,
        CASE WHEN support_tickets >= 5 THEN 1 ELSE 0 END AS heavy_support_user,
    
        premium_plan * log(num_logins + 1) AS premium_login_score,
    
        churn
    FROM users
    """

    df = con.execute(query).fetchdf()
    print(df.head())

    # Prepare training data
    X = df.drop("churn", axis=1)
    y = df["churn"]

    # Define preprocessing
    numeric_features = [
        "tenure_months",
        "num_logins",
        "support_tickets",
        "logins_per_month",
        "tickets_per_month",
        "premium_login_score"
    ]
    categorical_features = ["tenure_bucket"]
    binary_features = ["premium_plan", "is_high_login", "heavy_support_user"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("bin", "passthrough", binary_features)
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build pipeline and train
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Visualize churn by tenure bucket
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="tenure_bucket", hue="churn", palette="Set2")
    plt.title("Churn by Tenure Bucket")
    plt.xlabel("Tenure Group")
    plt.ylabel("User Count")
    plt.tight_layout()
    plt.show()
