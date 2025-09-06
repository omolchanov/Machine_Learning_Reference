import numpy as np
from statsmodels.stats.proportion import proportions_ztest

np.random.seed(42)

# --- Simulation parameters ---
num_users = 1000
p_A = 0.10  # 10% click rate for Ad A
p_B = 0.15  # 15% click rate for Ad B

# --- Simulate user interactions ---
group_A = np.random.binomial(1, p_A, size=num_users//2)
group_B = np.random.binomial(1, p_B, size=num_users//2)

# --- Calculate observed conversion rates ---
conv_rate_A = group_A.mean()
conv_rate_B = group_B.mean()

print(f"Observed conversion rate Ad A: {conv_rate_A:.3f}")
print(f"Observed conversion rate Ad B: {conv_rate_B:.3f}")

# --- Statistical significance test (two-proportion z-test) ---
successes = np.array([group_A.sum(), group_B.sum()])
trials = np.array([len(group_A), len(group_B)])

z_stat, p_value = proportions_ztest(successes, trials)
print(f"Z-statistic: {z_stat:.3f}, p-value: {p_value:.3f}")

if p_value < 0.05:
    print("Difference is statistically significant! Ad B is better.")
else:
    print("No statistically significant difference.")
