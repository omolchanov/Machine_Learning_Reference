"""
LSTM example for temporal propensity that is fully sequential, not flattened, and models a realistic user behavior.

Imagine an e-commerce website, where users have multiple sessions, and at each session we log:
- time_spent (minutes)
- pages_viewed
- added_to_cart (0/1)

We want to predict whether the user will purchase by the end of the sequence.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from sklearn.model_selection import train_test_split

# Configuration
np.set_printoptions(threshold=np.inf, suppress=True)

# --- 1. Generate synthetic sequential data ---
np.random.seed(42)
n_users = 1000
timesteps = 8  # each user has up to 8 sessions
features = 3   # time_spent, pages_viewed, added_to_cart

# Simulate user behavior
time_spent = np.random.uniform(1, 20, size=(n_users, timesteps, 1))  # minutes
pages_viewed = np.random.poisson(5, size=(n_users, timesteps, 1))     # pages
added_to_cart = np.random.binomial(1, 0.3, size=(n_users, timesteps, 1))  # 0/1

# Combine features
X = np.concatenate([time_spent, pages_viewed, added_to_cart], axis=2)

# Target: did the user purchase? Let's say:
# Purchase more likely if they added to cart multiple times and spent more time recently
score = (X[:, -3:, 0].sum(axis=1) * 0.05 +  # last 3 sessions time
         X[:, -3:, 1].sum(axis=1) * 0.02 +  # last 3 sessions pages
         X[:, -3:, 2].sum(axis=1) * 0.5)    # last 3 sessions cart
y = (score > 0.5).astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Build LSTM model ---
model = Sequential([
    Masking(mask_value=0.0, input_shape=(timesteps, features)),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

# --- 3. Train ---
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# --- 4. Predict propensity scores ---
propensity_scores = model.predict(X_test).flatten()

# --- 5. Convert to 0/1 predictions ---
y_pred = (propensity_scores >= 0.5).astype(int)

print("Example propensity scores:", propensity_scores[:10])
print("Binary predictions:", y_pred[:10])
print("Ground truth:", y_test[:10])
