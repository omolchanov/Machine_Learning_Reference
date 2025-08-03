import pandas as pd
import numpy as np
import random

# Configuration
n_records = 10_000
outlier_fraction = 0.01
null_fraction = 0.05

np.random.seed(42)
random.seed(42)

# Generate amount values with some outliers
def generate_amounts(n, outlier_fraction=0.01):
    n_outliers = int(n * outlier_fraction)
    n_normal = n - n_outliers

    normal_amounts = np.random.normal(loc=50000, scale=15000, size=n_normal)
    normal_amounts = np.clip(normal_amounts, 1000, None)
    outliers = np.random.randint(300000, 1000000, size=n_outliers)

    all_amounts = np.concatenate([normal_amounts, outliers])
    np.random.shuffle(all_amounts)
    return all_amounts.astype(float)

# Create base data
df = pd.DataFrame({
    "id": range(1, n_records + 1),
    "amount": generate_amounts(n_records, outlier_fraction),
    "churn": np.random.choice([0, 1], size=n_records, p=[0.8, 0.2]),
    "new_customer": np.random.choice([0, 1], size=n_records, p=[0.7, 0.3])
})

# Inject NaN values (~5% per column, excluding 'id')
for col in ['amount', 'churn', 'new_customer']:
    n_nulls = int(null_fraction * n_records)
    null_indices = np.random.choice(df.index, n_nulls, replace=False)
    df.loc[null_indices, col] = np.nan

# Save to CSV
df.to_csv("data/stagins_sales_df.csv", index=False)

