"""
Natural language processing (NLP) tasks frequently employ the Recurrent Neural Network (RNN) variant known
as Long Short-Term Memory (LSTM). RNNs are neural networks that process sequential data, such as time series
data or text written in a natural language. A particular kind of RNN called LSTMs can solve the issue
of vanishing gradients, which arises when traditional RNNs are trained on lengthy data sequences.

LSTM model for text classification

Guideline:
https://spotintelligence.com/2023/01/11/lstm-in-nlp-tasks/
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder

import netron


# The dataset with texts and labels for classification of the sentiment
texts = [
    "I'm so proud of you",
    "They don't care about me",
    "What's the difference"
]

labels = ['positive', 'negative', 'neutral']

# Hyperparameters
MAX_WORDS = 100  # max number of words to use in the vocabulary
MAX_LEN = 10  # max length of each text (in terms of number of words)
EMBEDDING_DIM = 100  # dimension of word embeddings
LSTM_UNITS = 64  # number of units in the LSTM layer

n_classes = len(labels)  # number of classes

# Tokenize the texts and create a vocabulary
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad the sequences so they all have the same length
X = pad_sequences(sequences, maxlen=MAX_LEN)

# Create one-hot encoded labels
labels = LabelEncoder().fit_transform(labels)
y = to_categorical(labels, n_classes)

# Build the model
model = Sequential([
    # Embedding layer maps input information from a high-dimensional to a lower-dimensional space,
    # allowing the network to learn more about the relationship between inputs and to process the data
    # more efficiently.
    # https://www.baeldung.com/cs/neural-nets-embedding-layers
    Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN, mask_zero=True),

    # https://stackoverflow.com/questions/64881855/kernel-and-recurrent-kernel-in-keras-lstms
    # https://stackoverflow.com/questions/55723284/what-is-the-meaning-of-multiple-kernels-in-keras-lstm-layer
    LSTM(LSTM_UNITS),
    Dense(n_classes, activation='softmax')
])

# Compile and train the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, batch_size=1, epochs=30)

MODEL_PATH = 'saved_models/lstm_model.keras'

model.save(MODEL_PATH)
netron.start(MODEL_PATH)
