import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np

# Step 1: Sample text data
sentences = [
    "I love cats",
    "You love dogs"
]

# Step 2: Tokenize the words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
word_index = tokenizer.word_index
print('Word Index:', word_index)
print('Sequences:', sequences)

# Step 3: Pad sequences to fixed length (adds 0 to the beginning of the sequence)
padded = pad_sequences(sequences, maxlen=4)
print("Padded Sequences:\n", padded)

vocab_size = len(word_index) + 1
print(vocab_size)
model = Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=3, input_length=4),
    layers.LayerNormalization()
])

output = model.predict(padded)
print(output)


