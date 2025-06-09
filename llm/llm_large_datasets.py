import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer

import numpy as np
from datasets import load_dataset


# === NN Parameters
DATASET_SIZE = 100000

BLOCK_SIZE = 128
BATCH_SIZE = 64
EMBED_DIMS = 64
NUM_HEADS = 2
FF_DIM = 128
EPOCHS = 2

DIRECTORY_PATH = 'models'


# === 1. Load and Prepare Dataset ===

# Load wikitext-2-raw-v1 from Hugging Face
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
print('The dataset is loaded')

# Combine all lines into one big string (truncate to 50k chars for speed)
raw_text = "\n".join(dataset["text"]).lower()[:DATASET_SIZE]


# === 2. Tokenize the data

# Character-level tokenizer
tokenizer = Tokenizer(char_level=True, filters='', lower=True)
tokenizer.fit_on_texts([raw_text])

vocab_size = len(tokenizer.word_index) + 1  # add 1 for padding index (0)
print('Vocalbulary size:', vocab_size)


def encode(s):
    return tokenizer.texts_to_sequences([s])[0]


def decode(l):
    return ''.join(tokenizer.sequences_to_texts([l])[0].split())


# Encode entire dataset
data = np.array(encode(raw_text), dtype=np.int32)
print('Encoded dataset shape:', data.shape[0])


def get_dataset(data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE):
    X, y = [], []
    for i in range(len(data) - block_size):
        X.append(data[i:i + block_size])
        y.append(data[i + 1:i + block_size + 1])
    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    return dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)


train_dataset = get_dataset(data)
print('Train dataset is ready')


# === 2. Transformer Block ===

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim=EMBED_DIMS, num_heads=NUM_HEADS, ff_dim=FF_DIM):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

    @tf.function
    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim
        })

        return config


# === 3. Build Model ===

def build_model(
        vocab_size,
        block_size=BLOCK_SIZE,
        embed_dim=EMBED_DIMS,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM
):
    inputs = layers.Input(shape=(block_size,))
    x = layers.Embedding(vocab_size, embed_dim)(inputs)

    # Add sinusoidal positional encoding manually
    positions = tf.range(start=0, limit=block_size, delta=1)
    position_embeddings = layers.Embedding(input_dim=block_size, output_dim=embed_dim)(positions)
    x = x + position_embeddings

    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.Dense(vocab_size)(x)
    return keras.Model(inputs, x)


model = build_model(vocab_size)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)
print('The model has been compiled')

# === 4. Train Model ===

model.fit(train_dataset, epochs=EPOCHS)
loss, acc = model.evaluate(train_dataset)
print(f"\nLoss: {loss:.3f}, Accuracy: {acc:.3f}")

model_id = f"llm_model_{random.randint(0, 1000)}_{acc:.3f}_{DATASET_SIZE}"
model.save(f"{DIRECTORY_PATH}/{model_id}")

print(f"The model has been saved to '/{DIRECTORY_PATH}/{model_id}'")

# === 6. Generate Text ===


def generate_text(
        start_text,
        model=model,
        tokenizer=tokenizer,
        block_size=BLOCK_SIZE,
        length=200,
        temperature=1.0
):
    input_ids = tokenizer.texts_to_sequences([start_text])[0]
    generated = input_ids[:]

    for _ in range(length):
        context = generated[-block_size:]
        if len(context) < block_size:
            context = [0] * (block_size - len(context)) + context
        x_input = tf.constant([context], dtype=tf.int32)
        logits = model(x_input, training=False)
        next_logits = logits[0, -1] / temperature
        next_id = tf.random.categorical(tf.expand_dims(next_logits, 0), num_samples=1).numpy()[0][0]
        generated.append(next_id)

    return decode(generated)


# === 7. Try Text Generation ===

prompt = "the future of ai"
output = generate_text(prompt)
print("\nGenerated Text:\n", output)
