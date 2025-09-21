"""
The PSI (Population Stability Index) is a metric commonly used in credit risk, marketing, and machine learning
to measure how much a variable’s distribution has shifted over time or between populations.
It’s particularly useful for monitoring model performance and detecting data drift.

Think of PSI as a health check for your data.

- You build a model on some old data (training set).
- Later, you use it on new data (production).
- If the new data looks very different from the old data, your model might give bad predictions.

PSI tells you how different the new data is from the old one.

Imagine you own a clothing store:

- Last year, 60% of customers bought shirts, 30% pants, 10% jackets.
- This year, 40% buy shirts, 20% pants, 40% jackets.

That’s a big shift in behavior. Your old “shirts and pants” strategy won’t work as well anymore.

PSI measures that shift in numbers:
- Small PSI → Customers behave almost the same → No worries.
- Medium PSI → Customers are changing → Keep an eye.
- Large PSI → Customers are very different now → Your old strategy (or model) is outdated.

How to read PSI

- PSI < 0.1 → Almost no change → Model is fine.
- 0.1 – 0.25 → Noticeable change → Monitor closely.
- > 0.25 → Big change → Model probably needs retraining.
"""
import numpy as np
import pandas as pd

# Data: old vs new distributions
ref = pd.DataFrame({"Category": ["Shirts", "Pants", "Jackets"], "Reference": [0.6, 0.3, 0.1]})
cur = pd.DataFrame({"Category": ["Shirts", "Pants", "Jackets"], "Current": [0.4, 0.2, 0.4]})


# PSI calculation
def psi(expected, actual):

    # avoid zeros
    expected = np.where(expected == 0, 0.0001, expected)
    actual = np.where(actual == 0, 0.0001, actual)
    return np.sum((actual - expected) * np.log(actual / expected))


psi_value = psi(ref["Reference"].values, cur["Current"].values)
print(f"PSI: {psi_value:.3f}")
