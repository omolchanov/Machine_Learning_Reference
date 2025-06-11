"""
To estimate the number of learning cycles and training time per epoch, we need to calculate the total iterations
required per epoch and approximate the duration based on empirical benchmarks.

Calculation Overview
    1. Steps Per Epoch = DATASET_SIZE / BATCH_SIZE
    2. Total Training Steps = Steps Per Epoch * EPOCHS
    3. Estimated Time Per Step (depends on hardware, but we can assume an approximate value)
    4. Total Estimated Training Time = Time Per Step * Total Training Steps
"""

import numpy as np

# Hyperparameters
BLOCK_SIZE = 128
BATCH_SIZE = 64
EMBED_DIMS = 64
NUM_HEADS = 2
FF_DIM = 128
EPOCHS = 2

if __name__ == '__main__':

    # Load the dataset and get its size
    encoded_data = np.load("data/dataset.npy")
    tokenized_length = len(encoded_data)

    # Compute steps per epoch
    num_samples = tokenized_length - BLOCK_SIZE
    steps_per_epoch = num_samples // BATCH_SIZE
    total_training_steps = steps_per_epoch * EPOCHS


    def adjust_time_per_step(initial_time, epochs):
        decay_factor = 0.75  # Example: 55% speedup per epoch
        times = [initial_time * (decay_factor ** epoch) for epoch in range(epochs)]
        return sum(times) / epochs  # Average time per step across epochs


    initial_time_per_step = 0.15
    adjusted_time_per_step = adjust_time_per_step(initial_time_per_step, EPOCHS)
    total_training_time = total_training_steps * adjusted_time_per_step

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_training_steps}")
    print(f"Approximate training time: {total_training_time:.0f} seconds | {total_training_time / 60:.1f} minutes")
