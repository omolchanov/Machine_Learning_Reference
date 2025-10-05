"""
Idea

When you want to understand the effect of some intervention (e.g., a new policy, marketing campaign, or treatment)
on a single unit (like a country, company, or individual), you need a counterfactual—what would have happened if the
intervention had not occurred.

The synthetic control method builds this counterfactual by creating a weighted combination of similar units that did
not receive the intervention. This combination is called the synthetic control.

Steps

- Select donor pool: Choose units (entities) that did not receive the intervention.
- Assign weights: Find weights for these donor units such that the weighted average of their pre-intervention outcomes
closely matches the treated unit's pre-intervention outcomes.
- Predict counterfactual: Use the weighted combination to estimate what the treated unit's outcome would have been
without the intervention.
- Estimate treatment effect: Compare the actual outcome of the treated unit after the intervention with the synthetic
control’s predicted outcome.
"""

"""
Imagine you want to study the effect of a smoking ban introduced in City A in year 6 on cigarette sales.

City A = treated unit
It experiences the intervention (smoking ban).

Cities B, C, D = donor pool
They do not implement the ban.
They are “control” units.

We observe sales data for all cities over 10 years (years 1–10).

Explanation

1. We simulate trends for a treated city and donor cities.
2. We fit weights for donor cities to best match the treated city before intervention.
3. We calculate a synthetic control series using the weights.
4. Plot the treated vs synthetic control to visualize the effect.
5. Compute the estimated treatment effect (difference between treated and synthetic control).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulate Data
np.random.seed(42)
years = np.arange(1, 11)
treated = 50 + np.arange(10) * 2  # treated city baseline trend
treated[5:] += 10  # policy effect starts at year 6

# donor cities
donor1 = 48 + np.arange(10) * 2 + np.random.normal(0, 1, 10)
donor2 = 52 + np.arange(10) * 2 + np.random.normal(0, 1, 10)
donor3 = 49 + np.arange(10) * 2 + np.random.normal(0, 1, 10)

df = pd.DataFrame({
    'Year': years,
    'Treated': treated,
    'Donor1': donor1,
    'Donor2': donor2,
    'Donor3': donor3
})

# Find synthetic control weights
# We'll fit weights to match the treated city BEFORE intervention (years 1-5)
X_pre = df.loc[df['Year'] <= 5, ['Donor1', 'Donor2', 'Donor3']].values
y_pre = df.loc[df['Year'] <= 5, 'Treated'].values

# Solve for weights using constrained least squares (weights sum to 1 and are non-negative)
from scipy.optimize import minimize


def loss(w):
    # predicted synthetic control
    y_hat = X_pre @ w
    return np.sum((y_pre - y_hat)**2)


# constraints: weights sum to 1, weights >= 0
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0, 1) for _ in range(X_pre.shape[1])]

res = minimize(loss, x0=[1/3, 1/3, 1/3], bounds=bounds, constraints=constraints)
weights = res.x
print("Synthetic control weights:", weights)

# ----------------------
# 3. Create synthetic control series
# ----------------------
X_full = df[['Donor1', 'Donor2', 'Donor3']].values
df['Synthetic'] = X_full @ weights

print(df)

# ----------------------
# 4. Plot results
# ----------------------
plt.figure(figsize=(10,6))
plt.plot(df['Year'], df['Treated'], label='Treated', marker='o')
plt.plot(df['Year'], df['Synthetic'], label='Synthetic Control', marker='x')
plt.axvline(5.5, color='gray', linestyle='--', label='Intervention')
plt.xlabel('Year')
plt.ylabel('Outcome')
plt.title('Synthetic Control Example')
plt.legend()
plt.show()

df['Effect'] = df['Treated'] - df['Synthetic']
print(df[['Year', 'Effect']])
