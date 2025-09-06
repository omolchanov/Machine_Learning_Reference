import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

np.set_printoptions(threshold=np.inf)

# --- Simulate user data ---
np.random.seed(42)
num_users = 1000

# Features: e.g., user behavior metrics
X = np.random.rand(num_users, 5)

# True probability of conversion
y_A = np.random.binomial(1, 0.05, size=num_users//2)
y_B = np.random.binomial(1, 0.25, size=num_users//2)

# Split features
X_A = X[:num_users//2]
X_B = X[num_users//2:]

# Combine for training
X_all = np.vstack([X_A, X_B])
y_all = np.concatenate([y_A, y_B])

# Neural network
model = Sequential([
    Dense(8, input_dim=5, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(0.01), loss='binary_crossentropy', metrics=["accuracy"])

# Train the model
model.fit(X_all, y_all, epochs=200, verbose=2)

# Evaluate each group
conversion_A = model.predict(X_A).mean()
conversion_B = model.predict(X_B).mean()

print(f"Predicted conversion rate Version A: {conversion_A:.3f}")
print(f"Predicted conversion rate Version B: {conversion_B:.3f}")

if conversion_B > conversion_A:
    print("Version B is better!")
else:
    print("Version A is better!")