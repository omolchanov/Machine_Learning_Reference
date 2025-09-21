"""
The Kolmogorov-Smirnov (K-S) metric is basically a way to measure how different two sets of numbers are.

Imagine this:
You have two groups of numbers (like test scores from two classes).
You make a graph that shows how many numbers are below each value — this is called the cumulative distribution.
The K-S metric asks: “What is the biggest gap between these two graphs?”
If the biggest gap is small, the two groups are very similar.
If the biggest gap is big, the two groups are quite different.
It’s like lining up the two sets of numbers on a number line and looking for the point where they are most different.

People use it to:
Check if a model’s predictions match reality.
Compare two datasets to see if they behave similarly.
Test if data follows a certain pattern (like a normal distribution).
"""

import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt


# Simulated predicted probabilities for a binary classifier
# Class 0 predictions
pred_class0 = np.random.beta(a=2, b=5, size=1000)  # skewed toward 0

# Class 1 predictions
pred_class1 = np.random.beta(a=5, b=2, size=1000)  # skewed toward 1


def plot_histograms():
    # Plot histograms of the two classes
    plt.figure(figsize=(8,5))
    plt.hist(pred_class0, bins=30, alpha=0.6, label="Class 0 predictions", density=True)
    plt.hist(pred_class1, bins=30, alpha=0.6, label="Class 1 predictions", density=True)

    # Add labels and title
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.title("Distributions of Class 0 vs Class 1 Predictions")
    plt.legend()
    plt.grid(True)

    plt.show()


# Run Kolmogorov–Smirnov test
ks_stat, p_value = ks_2samp(pred_class0, pred_class1)

"""
K-S = 0 → The two distributions are identical (no separation at all).
K-S close to 1 → The two distributions are very different (excellent separation).

Rule of thumb with K-S statistic:
KS > 0.6 → Very good separation.
0.4 < KS < 0.6 → Moderate separation.
KS < 0.2 → Poor separation (model almost guessing).
"""
print(f"K-S statistic: {ks_stat:.3f}")
print(f"P-value: {p_value:.3e}")

if ks_stat > 0.5:
    print("High K-S statistic: the model does a good job separating the classes.")
else:
    print("Low K-S statistic: the model struggles to separate the classes.")


# Plot cumulative distributions
def plot_cdf(data, label):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    plt.step(sorted_data, cdf, label=label)


plt.figure(figsize=(8,5))
plot_cdf(pred_class0, "Class 0 predictions")
plot_cdf(pred_class1, "Class 1 predictions")
plt.xlabel("Predicted probability")
plt.ylabel("Cumulative probability")
plt.title("K-S plot for classifier scores")
plt.legend()
plt.grid(True)
plt.show()
