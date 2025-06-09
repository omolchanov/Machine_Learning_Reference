"""
A Transformer-based model similar to GPT-style architecture, but on a small scale and character-level.
This script does:
- Encode text into tokens (characters)
- Build a mini Transformer model using Keras
- Train it to predict the next character
- Optionally generate text
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer

import numpy as np


# === 1. Prepare dataset ===

# Character-to-index mappings. Create encoder and fit
text = "hello world hello world  hello world"

base_tokenizer = Tokenizer(char_level=True)
base_tokenizer.fit_on_texts([text])

vocab_size = len(base_tokenizer.word_index) + 1
# print("Vocabulary size:", vocab_size)


# Encode string to integers
def encode(s):
    return base_tokenizer.texts_to_sequences([s])[0]


# Decode integers to string
def decode(l):
    # Handle zero-padding if present
    return ''.join(base_tokenizer.sequences_to_texts([l])[0].split())


# print(encode("hello"))
# print(decode([3, 4, 1, 1, 2]))


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


# Prepare features and labels
# feature -> piece of data sliced onto block size.
# label -> piece of data sliced onto block size + 1
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

    """
    The Embedding layer in Keras is used to convert integer-encoded categorical data (like words or tokens) into dense 
    vectors of fixed size. It's commonly used in NLP models to learn word representations during training.
    
    input_dim is the size of the vocabulary — i.e., the maximum integer index + 1 that can appear in your input data.
    Keras expects all token indices to be less than input_dim. So if your tokenizer assigns tokens 0 to 9 
    (10 unique tokens), input_dim should be 10, not 9.
    
    The output_dim in a Keras Embedding layer controls the dimensionality of the dense vector each word/token 
    is mapped to,
    """
    x = layers.Embedding(vocab_size, embed_dim)(inputs)

    """
    Positional Embedding Layer is  is used to add positional information to token embeddings — essential for models 
    like Transformers that don't have recurrence. Since Transformers don't process tokens in order (no recurrence), 
    we must manually add position information using positional embeddings.
    This is where AddPositionEmbs (or sine/cosine embeddings) come in — they inject sequence order into the model.
    """
    x = layers.AddPositionEmbs()(x) if hasattr(layers, 'AddPositionEmbs') else x  # fallback if not in TF

    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.Dense(vocab_size)(x)
    return keras.Model(inputs, x)


llm_model = build_model(vocab_size, block_size)

"""
SparseCategoricalCrossentropy is a loss function used for multi-class classification problems where:
- The output (prediction) is a probability distribution over classes.
- The target labels are integer class indices — not one-hot encoded.
Labels look like: [2, 0, 3, 1] (integers), not [[0, 0, 1, 0], ...].
"""
llm_model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="nadam",
    metrics=['accuracy']
)
# model.summary()


# === 4. Train and evaluate the Model ===
llm_model.fit(X_train, y_train, epochs=1000, verbose=0)
loss, acc = llm_model.evaluate(X_train, y_train)


# === 5. Generate Text ===
def generate_text(
        start_text,
        model=llm_model,
        tokenizer=base_tokenizer,
        length=10,
        block_size=block_size,
        temperature=1.0
):
    """
    Generate character-level text using a trained Transformer model.

    Args:
        model: Trained Keras model.
        tokenizer: Keras Tokenizer fitted on the char-level text.
        start_text (str): Initial seed text to begin generation.
        length (int): Number of characters to generate.
        block_size (int): Input sequence length model was trained on.
        temperature (float): Sampling randomness. Lower = more deterministic.

    Returns:
        str: Generated string.
    """
    # Encode the start text
    input_ids = tokenizer.texts_to_sequences([start_text])[0]
    generated = input_ids[:]

    for _ in range(length):
        # Get last `block_size` tokens (pad with 0 if too short)
        context = generated[-block_size:]

        if len(context) < block_size:
            context = [0] * (block_size - len(context)) + context

        # Shape to (1, block_size)
        x_input = tf.constant([context], dtype=tf.int32)

        # Predict next token logits
        logits = model(x_input, training=False)
        next_logits = logits[0, -1] / temperature

        # Sample next token ID
        next_id = tf.random.categorical(tf.expand_dims(next_logits, 0), num_samples=1).numpy()[0][0]

        # Append to the generated sequence
        generated.append(next_id)

    # Decode to string
    return ''.join(tokenizer.sequences_to_texts([generated])[0].split())


prompt = 'hello '
response = generate_text(str(prompt))
print("Generated text:", response)







