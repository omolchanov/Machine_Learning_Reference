"""
Imagine we are in e-commerce: users visit a website multiple times before deciding whether to purchase. At each visit,
we log things like:
- Time spent (minutes per session)
- Pages viewed
- Discount exposure (was a promo/discount shown)

We want to estimate the propensity to purchase, but with a Bayesian model so we also know the uncertainty
in our estimates.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from sklearn.model_selection import train_test_split

# --- 1. Generate synthetic sequential data ---
# Imagine we have 1000 users, each with up to 10 time steps (events).
# Each time step has 3 features, for example,
#   - time spent on site (normalized),
#   - number of pages viewed,
#   - whether a discount was shown (0/1).
n_users = 1000
timesteps = 10
features = 3

np.random.seed(42)
X = np.random.rand(n_users, timesteps, features)

# Target variable (did the user purchase?)
# Here we simulate that purchase probability depends mostly
# on the LAST event (e.g., last session before leaving the site).
y = (X[:, -1, 0] + X[:, -1, 1] * 0.5 + X[:, -1, 2] * 0.8 > 1.0).astype(int)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Build LSTM model ---
model = Sequential([
    # Masking layer allows us to ignore padded timesteps (if not all users have 10 events)
    Masking(mask_value=0.0, input_shape=(timesteps, features)),

    # LSTM captures temporal dependencies in user behavior sequences
    LSTM(64, return_sequences=False),
    Dropout(0.3),

    # Dense layers transform the sequence embedding into a probability
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")  # Output: probability of purchase
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])

# --- 3. Train the model ---
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1)

# --- 4. Predict propensity scores ---
propensity_scores = model.predict(X_test).flatten()
y_pred = (propensity_scores >= 0.5).astype(int)

print("Example propensity scores:", propensity_scores[:10])
print("Binary predictions:", y_pred[:10])
print("Ground truth:", y_test[:10])
