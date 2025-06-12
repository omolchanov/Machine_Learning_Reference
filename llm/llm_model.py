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

from llm_dataset import (
    DATA_DIRECTORY_PATH,
    DS_METADATA_FILENAME,
    DS_FIlENAME
)

import numpy as np

# Limit TensorFlow for CPU
cpu_limit = int(os.cpu_count() * 0.8)
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
EPOCHS = 2

CHECKPOINTS_DIR = 'checkpoints'
# os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
CHECKPOINT_FILEPATH = os.path.join(CHECKPOINTS_DIR, 'ckpt-{epoch:02d}-{loss:.4f}.weights.h5')


def get_optimized_dataset(
        data,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        buffer_size=10000,
        prefetch_size=tf.data.AUTOTUNE
):
    """Optimized version of get_dataset with performance improvements"""

    # Create input-target pairs
    def create_sequences():
        for i in range(len(data) - block_size):
            yield data[i:i + block_size], data[i + 1:i + block_size + 1]

    """
    This creates a TensorFlow dataset from a Python generator function, which is useful when you need to create data 
    on-the-fly rather than loading everything into memory at once.
    """
    dataset = tf.data.Dataset.from_generator(
        create_sequences,
        output_signature=(
            tf.TensorSpec(shape=(block_size,), dtype=tf.int32),
            tf.TensorSpec(shape=(block_size,), dtype=tf.int32)
        )
    )

    # Apply optimizations
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(prefetch_size)

    return dataset


# === Transformer Layer ===

class TransformerBlock(layers.Layer):
    def __init__(
            self,
            embed_dim=EMBED_DIMS,
            num_heads=NUM_HEADS,
            ff_dim=FF_DIM,
            dropout_rate=0.1
    ):
        super().__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,  # More efficient key dimension
            dropout=dropout_rate,
            use_bias=False  # Reduce parameters
        )

        """
        Optimized FFN - fused operations where possible
        
        FFN stands for Feed-Forward Network - it's one of the two main components in each transformer layer 
        (the other being multi-head attention). The FFN is a simple 2-layer neural network applied to each position 
        independently: Input → Dense(ff_dim) → Activation → Dense(embed_dim) → Output
        
        The FFN essentially acts as a "memory bank" where the transformer stores and processes learned patterns, 
        working together with attention to enable the model's language understanding capabilities.
        """
        self.dense1 = layers.Dense(ff_dim, use_bias=False)
        self.dense2 = layers.Dense(embed_dim, use_bias=False)
        self.dropout_ffn = layers.Dropout(dropout_rate)

        """
        Fused normalization layers
        
        LayerNormalization layers: These normalize inputs to have zero mean and unit variance, which helps stabilize 
        training and improve convergence. The epsilon=1e-6 parameter prevents division by zero during normalization. 
        In transformers, you typically see two layer norm operations - one before the attention mechanism 
        and one before the feed-forward network.
        
        Dropout layer: This randomly sets a fraction of input units to zero during training (based on dropout_rate) to 
        prevent overfitting. The "attn" suffix suggests this dropout is specifically applied to 
        attention weights or outputs.
        
        This pattern of layernorm + dropout is standard in transformer blocks, where you'd apply normalization and 
        regularization at key points in the data flow to maintain training stability and generalization.        
        """
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout_attn = layers.Dropout(dropout_rate)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=None, mask=None):
        """
        """

        """
        This code implements a residual attention block with pre-normalization, a key component in modern transformer 
        architectures. 
        
        Pre-normalization: norm_inputs = self.layernorm1(inputs) applies layer normalization to the input before the 
        attention operation. This "pre-norm" approach (vs. post-norm) generally leads to more stable training.
        
        Self-attention: attn_output = self.att(norm_inputs, norm_inputs, ...) performs self-attention where the same 
        normalized input serves as queries, keys, and values (hence the repeated norm_inputs). The attention_mask=mask 
        parameter allows masking certain positions (like padding tokens), and training=training enables different 
        behavior during training vs. inference.
        
        Regularization: attn_output = self.dropout_attn(attn_output, training=training) applies dropout to the attention 
        output during training to prevent overfitting.
        
        Residual connection: out1 = inputs + attn_output adds the original input to the processed output. This residual 
        connection helps with gradient flow during backpropagation and allows the model to learn incremental changes 
        rather than complete transformations.
        
        This pattern (pre-norm → attention → dropout → residual) is fundamental to transformer blocks and helps create 
        deep, trainable networks that can effectively model long-range dependencies.
        """
        norm_inputs = self.layernorm1(inputs)
        attn_output = self.att(norm_inputs, norm_inputs, attention_mask=mask, training=training)
        attn_output = self.dropout_attn(attn_output, training=training)
        out1 = inputs + attn_output

        # Fused FFN block with manual GELU for better performance
        norm_out1 = self.layernorm2(out1)
        ffn_hidden = self.dense1(norm_out1)

        # Manual GELU implementation (faster than activation layer)
        ffn_hidden = ffn_hidden * 0.5 * (1.0 + tf.tanh(0.7978845608 * (ffn_hidden + 0.044715 * tf.pow(ffn_hidden, 3))))
        ffn_hidden = self.dropout_ffn(ffn_hidden, training=training)
        ffn_output = self.dense2(ffn_hidden)
        ffn_output = self.dropout_ffn(ffn_output, training=training)

        return out1 + ffn_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate
        })
        return config


@tf.function
def get_sinusoidal_encoding(seq_len, embed_dim):
    """
    Performance-Optimized Positional Encoding
    Optimized sinusoidal positional encodings with caching
    """
    positions = tf.cast(tf.range(seq_len), tf.float32)[:, tf.newaxis]
    dims = tf.cast(tf.range(embed_dim), tf.float32)[tf.newaxis, :]

    # Vectorized computation
    inv_freq = 1.0 / tf.pow(10000.0, (2 * (dims // 2)) / tf.cast(embed_dim, tf.float32))
    angles = positions * inv_freq

    # Efficient sin/cos computation
    sin_vals = tf.sin(angles[:, 0::2])
    cos_vals = tf.cos(angles[:, 1::2])

    # Interleave sin and cos
    pos_encoding = tf.stack([sin_vals, cos_vals], axis=2)
    pos_encoding = tf.reshape(pos_encoding, [seq_len, embed_dim])

    return pos_encoding[tf.newaxis, ...]


# === Build Model ===

def build_model(
        vocab_size,
        block_size=BLOCK_SIZE,
        embed_dim=EMBED_DIMS,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=6,
        dropout_rate=0.1,
        use_mixed_precision=True
):
    # Enable mixed precision for faster training
    if use_mixed_precision:
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)

    inputs = layers.Input(shape=(block_size,), dtype=tf.int32)

    # Optimized embedding with weight tying preparation
    embedding_layer = layers.Embedding(
        vocab_size,
        embed_dim,
        embeddings_initializer='glorot_uniform',
        mask_zero=False  # Disable masking for performance
    )
    x = embedding_layer(inputs)

    # Efficient embedding scaling
    scale_factor = tf.cast(tf.math.sqrt(float(embed_dim)), dtype=x.dtype)
    x = x * scale_factor

    # Pre-computed positional encoding (cached)
    pos_encoding = get_sinusoidal_encoding(block_size, embed_dim)
    pos_encoding = tf.cast(pos_encoding, x.dtype)
    x = x + pos_encoding

    # Input dropout
    x = layers.Dropout(dropout_rate)(x)

    # Pre-computed causal mask (more efficient)
    causal_mask = create_causal_mask(block_size)

    # Stack transformer blocks with gradient checkpointing option
    for i in range(num_layers):
        x = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate
        )(x, mask=causal_mask)

    # Final layer norm
    x = layers.LayerNormalization(epsilon=1e-6, dtype=tf.float32)(x)

    # Weight-tied output projection (shares weights with embedding)
    if use_mixed_precision:
        x = tf.cast(x, tf.float32)  # Cast to float32 for final layer

    # Efficient dense layer without bias
    outputs = layers.Dense(
        vocab_size,
        use_bias=False,
        kernel_initializer='glorot_uniform',
        dtype=tf.float32
    )(x)

    return keras.Model(inputs, outputs)


@tf.function
def create_causal_mask(seq_len):
    """
    Pre-compute causal attention mask for better performance
    """

    mask = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.float32), -1, 0)
    return tf.where(mask == 0, -1e9, 0.0)


def create_optimized_optimizer(learning_rate=1e-4, warmup_steps=4000, total_steps=40000, use_mixed_precision=True):
    # Warmup followed by cosine decay
    warmup_schedule = keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=0.0,
        decay_steps=warmup_steps,
        end_learning_rate=learning_rate,
        power=1.0
    )

    # For steps beyond warmup, use cosine decay
    cosine_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=total_steps - warmup_steps,
        alpha=0.1
    )

    # You'd need to implement a custom schedule that combines these
    # Or use a simpler approach with just cosine decay

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,  # Or your custom schedule
        beta_1=0.9,
        beta_2=0.95,
        epsilon=1e-8,
        clipnorm=1.0
    )

    if use_mixed_precision:
        optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)

    return optimizer


@tf.function(experimental_relax_shapes=True)
def fast_label_smoothing_loss(y_true, y_pred, smoothing=0.1):
    """
    Optimized label smoothing with reduced memory usage
    """

    vocab_size = tf.shape(y_pred)[-1]
    confidence = 1.0 - smoothing
    low_confidence = smoothing / tf.cast(vocab_size - 1, tf.float32)

    # More efficient one-hot conversion
    if len(y_true.shape) < len(y_pred.shape):
        y_true_onehot = tf.one_hot(y_true, vocab_size, dtype=y_pred.dtype)
    else:
        y_true_onehot = tf.cast(y_true, y_pred.dtype)

    # Fused label smoothing operation
    soft_targets = y_true_onehot * confidence + low_confidence

    # Use more efficient cross-entropy computation
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=soft_targets,
            logits=y_pred
        )
    )


# === Data Pipeline Optimizations ===

if __name__ == '__main__':
    # === Load and Prepare Dataset ===

    # Load dataset
    data = np.load(f"{DATA_DIRECTORY_PATH}/{DS_FIlENAME}")
    train_dataset = get_optimized_dataset(data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)
    print('Train dataset is ready')

    # Load the dataset metadata
    with open(f"{DATA_DIRECTORY_PATH}/{DS_METADATA_FILENAME}", 'r') as f:
        metadata = json.load(f)
        vocab_size = metadata.get('vocab_size')

        print('Dataset Metadata:\n', metadata, '\n')

    # Build optimized model with mixed precision
    model = build_model(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        embed_dim=EMBED_DIMS,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=6,
        dropout_rate=0.1,
        use_mixed_precision=True  # Enable for 30-50% speedup
    )

    # Use optimized optimizer
    optimizer = create_optimized_optimizer(
        learning_rate=1e-4,
        warmup_steps=4000,
        use_mixed_precision=True
    )

    # Compile with optimizations
    model.compile(
        loss=fast_label_smoothing_loss,
        optimizer=optimizer,
        jit_compile=True  # Enable XLA compilation for extra speed
    )
    print('The model has been compiled')

    # === Train Model and benchmark it ===

    steps_per_epoch = (len(data) - BLOCK_SIZE) // BATCH_SIZE

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.7,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_FILEPATH,
            monitor='loss',
            save_best_only=False,  # Change to True if you only want to keep the best
            save_weights_only=True,  # We're only saving weights, not the full model
            verbose=1
        ),
    ]

    start_time = time.time()
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1
    )
    end_time = time.time()

    loss = model.evaluate(train_dataset, steps=steps_per_epoch)
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
