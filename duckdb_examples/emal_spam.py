"""
Email Spam Detection

Dataset: Email texts with labels (spam/ham)
Goal: Detect spam automatically
Tools: Bag-of-words or TF-IDF + classification model
"""

import duckdb
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


if __name__ == '__main__':

    # Create a sample dataset
    emails = pd.DataFrame({
        'email_id': list(range(1, 21)),
        'email_text': [
            # Spam examples
            "Win a brand new iPhone now! Click here!",
            "URGENT! You’ve won a lottery. Claim now.",
            "Cheap meds online, no prescription needed!",
            "Get rich quick! Work from home and earn $$$.",
            "Congratulations! You’ve been selected for a cash prize.",
            "Limited time offer! Buy 1 get 2 free!",
            "You’re pre-approved for a loan. Apply instantly.",
            "Exclusive deal on Rolex watches. Click now!",
            "Claim your reward points before they expire!",
            "Act now! Your account is about to be suspended.",

            # Ham examples
            "Hey, are we still on for lunch tomorrow?",
            "Meeting rescheduled to 3 PM. Please confirm.",
            "Don't forget the weekly team sync-up.",
            "Can you review the report before the meeting?",
            "Happy birthday! Hope you have a great day.",
            "Please find attached the latest project update.",
            "Let’s schedule a call to discuss next steps.",
            "Thanks for your feedback on the draft.",
            "Looking forward to our trip next weekend!",
            "Reminder: your dentist appointment is at 9 AM."
        ],
        'label': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 10 spam
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 10 ham
    })

    # Connect to DuckDB and register the data
    con = duckdb.connect()
    con.register("emails", emails)

    # Use DuckDB to preprocess, tokenize, and get cleaned text
    query = """
    WITH cleaned AS (
        SELECT 
            email_id,
            label,
            regexp_replace(lower(email_text), '[^a-z0-9 ]', '', 'g') AS clean_text
        FROM emails
    ),
    tokens AS (
        SELECT 
            email_id,
            label,
            UNNEST(string_split(clean_text, ' ')) AS token
        FROM cleaned
    ),
    filtered AS (
        SELECT token
        FROM tokens
        GROUP BY token
        HAVING COUNT(*) > 1  -- filter rare tokens
    ),
    final_text AS (
        SELECT 
            t.email_id,
            t.label,
            STRING_AGG(t.token, ' ') AS filtered_text
        FROM tokens t
        INNER JOIN filtered f ON t.token = f.token
        GROUP BY t.email_id, t.label
    )
    SELECT email_id, label, filtered_text FROM final_text
    ORDER BY email_id
    """
    df = con.execute(query).fetchdf()

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["filtered_text"])
    y = df["label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Train a model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    # Predict new email
    import re

    new_email = ["Hello. Lunch ?"]
    cleaned = re.sub(r'[^a-z0-9 ]', '', new_email[0].lower())
    new_vec = vectorizer.transform([cleaned])

    prediction = model.predict(new_vec)[0]
    print(f"New email classified as: {'Spam' if prediction == 1 else 'Ham'}")
