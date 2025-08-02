"""
Sentiment Analysis

Dataset: Product reviews
Goal: Classify sentiment as positive/negative
Tools: Naive Bayes / Logistic Regression + basic NLP
"""

import duckdb
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)

if __name__ == '__main__':

    # Sample product review dataset
    data = pd.DataFrame({
        'review_id': [1, 2, 3, 4, 5, 6],
        'review_text': [
            "Amazing product, really loved it!",
            "Terrible quality. Broke after one use.",
            "Great value for money",
            "Worst product I ever bought",
            "Pretty decent, works as expected",
            "Do not buy this, complete waste of money"
        ],
        'label': [1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative
    })

    # Register in DuckDB
    con = duckdb.connect()
    con.register("reviews", data)

    # Run query with preprocessing inside SQL
    query = """
        SELECT 
            regexp_replace(LOWER(review_text), '[^a-z0-9 ]', '', 'g') AS clean_text,
            label
        FROM reviews
        WHERE LENGTH(review_text) > 10
    """
    df = con.execute(query).fetchdf()

    # Vectorize text
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train logistic regression
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.2f}")

    # Predict on new review
    new_review = ["Do not buy this"]

    # Preprocess new review the same way (lowercase + remove punctuation)
    import re

    cleaned_review = [re.sub(r'[^a-z0-9 ]', '', new_review[0].lower())]
    new_vec = vectorizer.transform(cleaned_review)
    prediction = model.predict(new_vec)[0]

    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f"\nReview: \"{new_review[0]}\"\nSentiment: {sentiment}")
