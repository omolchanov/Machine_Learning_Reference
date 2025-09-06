import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

# --- 1. Generate synthetic dataset ---
np.random.seed(42)

n_samples = 500
n_features = 3

# Random input features
X = np.random.rand(n_samples, n_features)

# True relationship: y = 2*x0 - 3*x1 + 5*x2 + 1 + noise
true_weights = np.array([2, -3, 5])
true_bias = 1
noise = np.random.randn(n_samples) * 0.5

y = X @ true_weights + true_bias + noise

# --- 2. Train Linear Regression ---
model = LinearRegression()
model.fit(X, y)

print("Learned coefficients:", model.coef_)
print("Learned intercept:", model.intercept_)

# --- 3. Predictions ---
y_pred = model.predict(X)


print("R2 score:", r2_score(y, y_pred))

# --- 4. Plot Predictions vs Ground Truth ---
plt.figure(figsize=(7, 7))
plt.scatter(y, y_pred, alpha=0.6, edgecolor="k")
plt.xlabel("True values", fontsize=12)
plt.ylabel("Predicted values", fontsize=12)
plt.title("Predictions vs True values", fontsize=14)

# Ideal line (y = x)
lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
plt.plot(lims, lims, 'r--', label="Ideal fit")

plt.legend()
plt.grid(True)
# plt.show()

# --- 5. Plot Feature Weights as Bar Chart ---
plt.figure(figsize=(6, 4))
plt.bar(range(len(model.coef_)), model.coef_, color="skyblue", edgecolor="k")
plt.xlabel("Feature index", fontsize=12)
plt.ylabel("Coefficient value", fontsize=12)
plt.title("Feature weights (coefficients)", fontsize=14)
plt.axhline(0, color="black", linewidth=0.8)
# plt.show()

# --- 6. Plot residuals
residuals = y - y_pred
plt.hist(residuals, bins=20)
plt.title("Residual distribution")
# plt.show()

# --- 7. Permutation Importance (model-agnostic feature importance) ---
"""
Permutation importance plot → How much each feature matters by shuffling it and measuring performance drop 
(model-agnostic, more robust than raw coefficients).

If a feature has a high bar → the model heavily relies on it.
If a feature has a small bar → shuffling it didn’t change much, so the model doesn’t depend strongly on it.
If a bar is negative or near zero → the feature may be noise or irrelevant.
"""

result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
print("Permutation importances:", result.importances_mean)

plt.figure(figsize=(6, 4))
plt.bar(range(len(result.importances_mean)), result.importances_mean,
        yerr=result.importances_std, color="lightgreen", edgecolor="k")
plt.xlabel("Feature index", fontsize=12)
plt.ylabel("Importance (mean decrease in score)", fontsize=12)
plt.title("Permutation Feature Importance", fontsize=14)
plt.show()
