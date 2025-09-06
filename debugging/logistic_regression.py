import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

# --- 1. Generate synthetic classification dataset ---
X, y = make_classification(
    n_samples=500,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

feature_names = [f"feature_{i}" for i in range(X.shape[1])]

# --- 2. Train Logistic Regression ---
model = LogisticRegression()
model.fit(X, y)

print("Learned coefficients:", model.coef_)
print("Learned intercept:", model.intercept_)

# --- 3. Predictions ---
y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)
print("Training accuracy:", acc)

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y, y_pred)
plt.title("Confusion Matrix (Logistic Regression)")
plt.show()

# --- 4. Plot Feature Weights as Bar Chart ---
plt.figure(figsize=(6, 4))
plt.bar(feature_names, model.coef_[0], color="skyblue", edgecolor="k")
plt.xlabel("Feature")
plt.ylabel("Coefficient value")
plt.title("Feature weights (coefficients)", fontsize=14)
plt.axhline(0, color="black", linewidth=0.8)
plt.show()

# --- 5. Permutation Importance ---
result = permutation_importance(model, X, y, n_repeats=10, random_state=42)

print("\nPermutation Feature Importances:")
for name, importance in zip(feature_names, result.importances_mean):
    print(f"  {name}: {importance:.4f}")

plt.figure(figsize=(6, 4))
plt.bar(feature_names, result.importances_mean,
        yerr=result.importances_std, color="lightgreen", edgecolor="k")
plt.xlabel("Feature")
plt.ylabel("Importance (mean decrease in accuracy)")
plt.title("Permutation Feature Importance", fontsize=14)
plt.show()
