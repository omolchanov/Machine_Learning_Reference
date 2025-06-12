import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import random
import json
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from llm_tokenizer import (
    DATA_DIRECTORY_PATH,
    DS_METADATA_FILENAME,
    DS_FIlENAME
)

import numpy as np

# Limit TensorFlow for CPU
cpu_limit = int(os.cpu_count() * 0.5)
tf.config.threading.set_intra_op_parallelism_threads(cpu_limit)
tf.config.threading.set_inter_op_parallelism_threads(cpu_limit)


MODELS_DIRECTORY_PATH = 'models'
MODEL_METADATA_FILENAME = 'model_metadata.json'

# === NN Parameters
BLOCK_SIZE = 128
BATCH_SIZE = 64
EMBED_DIMS = 64
NUM_HEADS = 2
FF_DIM = 128
EPOCHS = 5


def get_dataset(data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE):
    data = tf.convert_to_tensor(data, dtype=tf.int32)

    # Create a dataset from the flat data array
    ds = tf.data.Dataset.from_tensor_slices(data)

    # Use window to create (X, y) pairs lazily and efficiently
    ds = ds.window(block_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(block_size + 1))

    # Split into input (X) and target (y)
    ds = ds.map(lambda window: (window[:-1], window[1:]), num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle, batch, and prefetch
    return ds.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# === Transformer Layer ===

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim=EMBED_DIMS, num_heads=NUM_HEADS, ff_dim=FF_DIM, dropout_rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=dropout_rate
        )
        # Improved FFN with GELU activation and dropout
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
            layers.Dropout(dropout_rate)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

    @tf.function
    def call(self, inputs, training=None, mask=None):
        # Pre-norm architecture (modern approach)
        norm_inputs = self.layernorm1(inputs)
        attn_output = self.att(norm_inputs, norm_inputs, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output  # Residual connection

        norm_out1 = self.layernorm2(out1)
        ffn_output = self.ffn(norm_out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return out1 + ffn_output  # Residual connection

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate
        })
        return config


def get_sinusoidal_encoding(seq_len, embed_dim):
    """Generate sinusoidal positional encodings"""
    positions = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
    dims = tf.range(embed_dim, dtype=tf.float32)[tf.newaxis, :]

    angles = positions / tf.pow(10000.0, (2 * (dims // 2)) / tf.cast(embed_dim, tf.float32))

    # Apply sin to even indices, cos to odd indices
    pos_encoding = tf.where(
        tf.cast(dims % 2, tf.bool),
        tf.cos(angles),
        tf.sin(angles)
    )

    return pos_encoding[tf.newaxis, ...]


# === Build Model ===

def build_model(
        vocab_size,
        block_size=BLOCK_SIZE,
        embed_dim=EMBED_DIMS,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=6,  # Multiple transformer layers
        dropout_rate=0.1
):
    inputs = layers.Input(shape=(block_size,))

    # Token embeddings with scaling
    x = layers.Embedding(vocab_size, embed_dim)(inputs)
    x = x * tf.math.sqrt(tf.cast(embed_dim, tf.float32))  # Scale embeddings

    # Add sinusoidal positional encoding
    pos_encoding = get_sinusoidal_encoding(block_size, embed_dim)
    x = x + pos_encoding

    # Input dropout
    x = layers.Dropout(dropout_rate)(x)

    # Create causal mask for autoregressive generation
    causal_mask = tf.linalg.band_part(tf.ones((block_size, block_size)), -1, 0)
    causal_mask = tf.where(causal_mask == 0, -1e9, 0.0)

    # Stack multiple transformer blocks
    for i in range(num_layers):
        x = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate
        )(x, mask=causal_mask)

    # Final layer norm
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Output projection
    outputs = layers.Dense(vocab_size, use_bias=False)(x)

    return keras.Model(inputs, outputs)


def create_optimizer_with_warmup(learning_rate=1e-4, warmup_steps=4000, embed_dim=EMBED_DIMS):
    """Create optimizer with learning rate warmup schedule"""
    lr_schedule = keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=warmup_steps,
        end_learning_rate=learning_rate * 0.1,
        power=0.5
    )

    return keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )


def label_smoothing_loss(y_true, y_pred, smoothing=0.1):
    """Label smoothing for better generalization - works with sparse integer labels"""
    vocab_size = tf.cast(tf.shape(y_pred)[-1], tf.float32)
    confidence = 1.0 - smoothing
    low_confidence = smoothing / (vocab_size - 1.0)

    # Convert sparse labels to one-hot if needed
    if len(y_true.shape) < len(y_pred.shape):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.cast(vocab_size, tf.int32))

    # Apply label smoothing
    y_true = tf.cast(y_true, tf.float32)
    soft_targets = y_true * confidence + (1.0 - y_true) * low_confidence

    return keras.losses.categorical_crossentropy(soft_targets, y_pred, from_logits=True)


if __name__ == '__main__':
    # === Load and Prepare Dataset ===

    # Load dataset
    data = np.load(f"{DATA_DIRECTORY_PATH}/{DS_FIlENAME}")
    train_dataset = get_dataset(data)
    print('Train dataset is ready')

    # Load the dataset metadata
    with open(f"{DATA_DIRECTORY_PATH}/{DS_METADATA_FILENAME}", 'r') as f:
        metadata = json.load(f)
        vocab_size = metadata.get('vocab_size')

        print('Dataset Metadata:\n', metadata, '\n')

    model = build_model(vocab_size)
    optimizer = create_optimizer_with_warmup(learning_rate=1e-4, warmup_steps=4000)

    model.compile(
        loss=label_smoothing_loss,
        optimizer=optimizer
    )
    print('The model has been compiled')

    # === Train Model and benchmark it ===

    steps_per_epoch = (len(data) - BLOCK_SIZE) // BATCH_SIZE

    start_time = time.time()
    model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=steps_per_epoch)
    end_time = time.time()

    loss = model.evaluate(train_dataset)
    training_duration = end_time - start_time

    n_params = model.count_params()

    print(f"\nLoss: {loss:.3f}")
    print(f"Total training time: {training_duration:.2f} seconds")
    print(f"Number of parameters: {n_params}")

    # === Save Model ===

    dataset_size = metadata.get('dataset_size')

    model_id = f"llm_model_{random.randint(0, 1000)}_{dataset_size}"
    model_pathname = f"{MODELS_DIRECTORY_PATH}/{model_id}"
    model.save(model_pathname)

    print(f"\nThe model has been saved to {model_pathname}")

    # === Save Model Metadata ===

    metadata = {
        'block_size': BLOCK_SIZE,
        'loss': loss,
        'training_duration': training_duration,
        'dataset_size': dataset_size,
        'n_parameters': n_params
    }

    metadata_pathname = f"{model_pathname}/{MODEL_METADATA_FILENAME}"
    with open(metadata_pathname, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"The model metadata has been saved to {metadata_pathname}")
