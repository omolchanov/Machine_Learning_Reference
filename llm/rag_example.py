"""
1) External retrieval: fetch Wikipedia intro paragraph for the query
2) Tokenizer: convert text data into integer sequences for Keras
3) Toy dataset: small training examples + dummy labels
4) Sequential model: Embedding + LSTM + Dense for simple classification
5) Model training: fit model on the tiny dataset
6) RAG query function:
     a) fetch Wikipedia text
     b) concatenate query + retrieved snippet
     c) tokenize + pad sequence
     d) run model prediction
     e) take argmax to get predicted label
7) Example run: input a query, retrieve snippet, predict label
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import requests

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

import numpy as np


# External retrieval (Wikipedia)
def fetch_wikipedia_summary(title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": title,
    }
    headers = {
        "User-Agent": "MinimalRAGBot/0.1 (https://example.com)"
    }
    r = requests.get(url, params=params, headers=headers, timeout=10)
    r.raise_for_status()  # will raise an error if status != 200
    data = r.json()
    pages = data["query"]["pages"]
    for pid, page in pages.items():
        if pid == "-1":
            return None
        return page.get("extract", "")
    return None

# title = fetch_wikipedia_summary("Paris")
# print(title[:400])


# Tokenizer + toy dataset
texts = [
    "Paris is the capital of France.",
    "Mount Everest is the tallest mountain.",
    "PHP is a programming language."
]

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
vocab_size = len(tokenizer.word_index) + 1

X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, padding="post")
y = np.array([0, 1, 2])

# Minimal Sequential NN
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=16, input_length=X.shape[1]),
    LSTM(16),
    Dense(len(set(y)), activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=150, verbose=2)


# RAG-like query pipeline
def rag_query(query):

    # fetch external info
    wiki_text = fetch_wikipedia_summary(query)
    if not wiki_text:
        wiki_text = ""
    augmented = query + " " + wiki_text

    # tokenize + pad
    seq = tokenizer.texts_to_sequences([augmented])
    seq = pad_sequences(seq, maxlen=X.shape[1], padding="post")

    # run model
    pred = model.predict(seq, verbose=0)
    label = np.argmax(pred, axis=1)[0]

    return {
        "query": query,
        "retrieved": wiki_text[:200] + "...",
        "prediction": int(label)
    }


"""
You entered the query "Python".
The code called Wikipedia’s API and retrieved the intro snippet from the Python (disambiguation) page.
That text was concatenated with your query, tokenized, and fed into your tiny Keras model.
The model produced a probability distribution over the 3 dummy labels [0, 1, 2] that we defined in the training step.
The highest-probability class was 2, so you got Model prediction: 2.
"""
if __name__ == "__main__":
    result = rag_query("PHP")
    print("Query:", result["query"])
    print("Retrieved snippet:", result["retrieved"])
    print("Model prediction:", result["prediction"])
