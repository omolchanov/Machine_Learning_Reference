"""
Compact Transformer Training Time Calculator
Simple estimation for your optimized transformer model
"""

import numpy as np
from llm_dataset import LlmDataset


# Hyperparameters
BLOCK_SIZE = 128
BATCH_SIZE = 64
EMBED_DIMS = 64
NUM_HEADS = 2
FF_DIM = 128
EPOCHS = 1


def get_vocab_size(data):
    """Load vocab size from metadata, fallback to estimation"""
    return len(np.unique(data))


def estimate_training_time(vocab_size, dataset_size, optimized=True):
    """Simple time estimation based on model size and optimizations"""

    # Base time per step (seconds) - calibrated from real results
    base_time = 0.8  # Adjusted based on 65s real vs 228s predicted

    # Adjust for model complexity
    vocab_factor = max(1.0, (vocab_size / 10000) ** 0.5)  # Square root scaling
    embed_factor = (EMBED_DIMS * FF_DIM) / (64 * 128)

    complexity_factor = vocab_factor * embed_factor
    base_time *= complexity_factor

    # Apply optimization speedups if using optimized model
    if optimized:
        base_time /= 2  # Minimal speedup factor

    # Calculate steps and total time
    steps_per_epoch = (dataset_size - BLOCK_SIZE) // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS
    total_time = total_steps * base_time

    # Minimal overhead
    total_time *= 1.1

    return total_time, steps_per_epoch, total_steps


if __name__ == '__main__':
    data = LlmDataset.load_ds()
    vocab_size = get_vocab_size(data)

    print(f"Dataset size: {len(data):,} tokens")
    print(f"Vocab size: {vocab_size:,}")

    # Calculate training time
    total_time, steps_per_epoch, total_steps = estimate_training_time(
        vocab_size, len(data), optimized=True
    )

    print(f"\nSteps per epoch: {steps_per_epoch:,}")
    print(f"Total steps: {total_steps:,}")
    print(f"\nEstimated training time:")
    print(f"  {total_time:.0f} seconds")
    print(f"  {total_time / 60:.1f} minutes")
    print(f"  {total_time / 3600:.2f} hours")