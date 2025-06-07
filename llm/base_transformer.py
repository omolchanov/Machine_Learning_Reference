"""
A Transformer-based model similar to GPT-style architecture, but on a small scale and character-level.
This script does:
- Encode text into tokens (characters)
- Build a mini Transformer model using Keras
- Train it to predict the next character
- Optionally generate text
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from sklearn.preprocessing import LabelEncoder

# === 1. Prepare dataset ===
text = "hello world hello world"
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Character-to-index mappings. Create encoder and fit
encoder = LabelEncoder()
enc = encoder.fit(list(chars))


# Encode string to integers
def encode(s):
    return encoder.transform(list(s))


# Decode integers to string
def decode(l):
    return ''.join(encoder.inverse_transform(l))


# print(encode(text))
# print(decode([3, 2, 4, 4, 5, 0, 7, 5, 6, 4, 1, 0, 3, 2, 4, 4, 5, 0, 7, 5, 6, 4, 1]))

# Tokenized data
data = np.array(encode(text), dtype=np.int32)
# print(data)
# print(decode(data))

# Hyperparameters
block_size = 4  # context window
batch_size = 8


def get_batches(data, block_size, batch_size):
    """
    Takes a sequence of encoded tokens (integers representing characters) and splitting it into (input, target) pairs.
    Each pair is a small chunk of the sequence, where:
    - input = current sequence of tokens
    - target = next sequence of tokens (shifted by one)

    This is used to train a model to predict the next character.
    """

    X, y = [], []
    for i in range(len(data) - block_size):
        X.append(data[i:i+block_size])
        y.append(data[i+1:i+block_size+1])
    X = np.array(X[:batch_size])
    y = np.array(y[:batch_size])
    return X, y


X_train, y_train = get_batches(data, block_size, batch_size)

# print(X_train)
# print('=======================')
# print(y_train)

# === 2. Build Transformer Block ===


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()

        """
        MultiHeadAttention is a layer designed to implement the multi-head attention mechanism popularized by the
        Transformer architecture (from the famous "Attention is All You Need" paper).
        Multi-head attention allows the model to jointly attend to information from different representation
        subspaces at different positions. Instead of performing a single attention function, it splits
        the queries, keys, and values into multiple heads, performs attention on each separately,
        and then concatenates the results.
        This helps the model capture different types of relationships and patterns in the input data more effectively.
        """
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])

        """
        Layer Normalization normalizes each sample independently, calculating mean and variance over the 
        features/channels of that sample.
        Why use LayerNormalization?
        - helps stabilize and speed up training.
        - keeps mean close to 0 and variance close to 1 for each feature vector.
        - commonly used in Transformer models and many NLP architectures.
        
        epsilon is a tiny constant added inside the square root to avoid division by zero or very 
        small numbers that cause numerical instability.        
        """
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)


# === 3. Create Mini-LLM Model ===

def build_model(vocab_size, block_size, embed_dim=32, num_heads=2, ff_dim=64):
    inputs = layers.Input(shape=(block_size,))
    x = layers.Embedding(vocab_size, embed_dim)(inputs)
    x = layers.AddPositionEmbs()(x) if hasattr(layers, 'AddPositionEmbs') else x  # fallback if not in TF
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.Dense(vocab_size)(x)
    return keras.Model(inputs, x)


model = build_model(vocab_size, block_size)
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer="adam",  metrics=['accuracy'])
model.summary()


# === 4. Train and evaluate the Model ===
model.fit(X_train, y_train, epochs=100, verbose=1)
loss, acc = model.evaluate(X_train, y_train)


# === 5. Generate Text ===

def generate_text(model, start_text="", length=20):
    input_eval = list(encode(start_text)[-block_size:])  # make sure it's a list
    for _ in range(length):
        x = np.array([input_eval[-block_size:]])
        preds = model.predict(x, verbose=0)[0][-1]
        next_id = tf.random.categorical(tf.expand_dims(preds, 0), num_samples=1).numpy()[0][0]
        input_eval.append(next_id)
    return decode(input_eval)


print(generate_text(model, "hello w"))







