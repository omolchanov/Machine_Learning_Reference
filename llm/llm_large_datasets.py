import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import random
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from llm_tokenizer import (
    DATA_DIRECTORY_PATH,
    DS_METADATA_FILENAME,
    DS_FIlENAME
)

import numpy as np


MODELS_DIRECTORY_PATH = 'models'
MODEL_METADATA_FILENAME = 'model_metadata.json'

# === NN Parameters
BLOCK_SIZE = 128
BATCH_SIZE = 64
EMBED_DIMS = 64
NUM_HEADS = 2
FF_DIM = 128
EPOCHS = 1


def get_dataset(data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE):
    X, y = [], []
    for i in range(len(data) - block_size):
        X.append(data[i:i + block_size])
        y.append(data[i + 1:i + block_size + 1])
    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    return dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# === Transformer Layer ===

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


# === Build Model ===

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


if __name__ == '__main__':
    # === Load and Prepare Dataset ===

    # Load dataset
    data = np.load(f"{DATA_DIRECTORY_PATH}/{DS_FIlENAME}")

    # Load the metadata
    with open(f"{DATA_DIRECTORY_PATH}/{DS_METADATA_FILENAME}", 'r') as f:
        metadata = json.load(f)

    print('Dataset Metadata:\n', metadata, '\n')

    train_dataset = get_dataset(data)
    print('Train dataset is ready')

    vocab_size = metadata.get('vocab_size')
    print('Vocalbulary size:', vocab_size, '\n')

    model = build_model(vocab_size)

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy']
    )
    print('The model has been compiled')

    # === Train Model ===

    model.fit(train_dataset, epochs=EPOCHS)
    loss, acc = model.evaluate(train_dataset)
    print(f"\nLoss: {loss:.3f}, Accuracy: {acc:.3f}")

    # === Save Model ===

    dataset_size = metadata.get('dataset_size')

    model_id = f"llm_model_{random.randint(0, 1000)}_{acc:.3f}_{dataset_size}"
    model_pathname = f"{MODELS_DIRECTORY_PATH}/{model_id}"
    model.save(model_pathname)

    print(f"\nThe model has been saved to {model_pathname}")

    # === Save Model Metadata ===

    metadata = {
        'block_size': BLOCK_SIZE
    }

    metadata_pathname = f"{model_pathname}/{MODEL_METADATA_FILENAME}"
    with open(metadata_pathname, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"The model metadata has been saved to {metadata_pathname}")
