"""
A propensity score is a concept from statistics and causal inference.
In simple terms, it’s the probability that a unit (person, patient, customer, etc.) receives a treatment given their
observed characteristics.

Formally:
e(x)=P(T=1∣X=x)

T = treatment indicator (1 if treated, 0 if not)
X = vector of observed covariates (age, income, health status, etc.)

In observational studies, unlike randomized trials, people are not randomly assigned to treatment vs. control.
That creates selection bias: maybe younger people are more likely to get the treatment, so comparing treated
vs. untreated directly is unfair.

The propensity score helps by:
- Balancing groups: If two groups have the same distribution of propensity scores, their covariates are (on average)
balanced, like in a randomized experiment.
- Reducing bias: Instead of comparing raw treated vs. untreated groups, we compare groups matched/weighted/stratified
by their propensity scores.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt

# Configuration
np.set_printoptions(threshold=np.inf, suppress=True)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)

# Generate synthetic data
np.random.seed(42)
n = 500

# Covariate: age
age = np.random.normal(40, 12, n)

"""
Treatment assignment depends on age
This makes older people more likely to get treatment.
People below 40 have a low probability.
People around 40 have a ~50/50 chance.
People above 40 are very likely.
"""
treatment = np.random.binomial(1, p=1/(1+np.exp(-(age-40)/5)))

"""
This defines how the outcome variable is generated in the synthetic dataset.

So the model is basically:
Y=2⋅T+0.1⋅Age+ε

Y = outcome
T = treatment (0 or 1)
N(0,1) = noise

This line creates a synthetic dataset that mimics a real-world problem where:
- Treatment is not randomly assigned
- Confounders affect both treatment and outcome
"""
outcome = 2*treatment + 0.1*age + np.random.normal(0, 1, n)

df = pd.DataFrame({"age": age.round(1), "treatment": treatment, "outcome": outcome})
# print(df)

"""
The naive difference in means is the simplest way to estimate a treatment effect — just compare the average outcome 
in the treated group vs. the untreated group.

In our synthetic example:
- Older people are more likely to get treated.
- Older people also naturally have higher outcomes (age → outcome).
"""
print("Naive difference in means:")
print(df[df.treatment == 1].outcome.mean() - df[df.treatment == 0].outcome.mean())

# Estimate propensity scores (logistic regression)
X = df[["age"]]
y = df["treatment"]

logit = LogisticRegression()
logit.fit(X, y)
df["propensity"] = logit.predict_proba(X)[:, 1]

"""
Interpretation:
- High propensity score (close to 1) → This unit was very likely to get treated based on its characteristics.
- Low propensity score (close to 0) → This unit was very unlikely to get treated.
- Around 0.5 → Treatment assignment was uncertain; both treated and untreated are possible.
"""
print(df.head())

# Matching (nearest neighbor on propensity score) ---
treated = df[df.treatment == 1]
control = df[df.treatment == 0]

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[["propensity"]])

"""
Each treated unit with a control unit that has the closest propensity score.
Treated group → query points.
Control group → database of possible neighbors.
"""
distances, indices = nn.kneighbors(treated[["propensity"]])
matched_controls = control.iloc[indices.flatten()]
matched_controls = matched_controls.set_index(treated.index)

# Scatter plots
plt.scatter(control["propensity"], control["outcome"], alpha=0.5, label="Control (untreated)", color="blue")
plt.scatter(treated["propensity"], treated["outcome"], alpha=0.7, label="Treated", color="red")

# Draw lines for matches
for i, treated_unit in treated.iterrows():
    matched_unit = matched_controls.loc[i]
    plt.plot([treated_unit["propensity"], matched_unit["propensity"]],
             [treated_unit["outcome"], matched_unit["outcome"]],
             color="gray", alpha=0.5)

plt.xlabel("Propensity Score")
plt.ylabel("Outcome")
plt.title("Propensity Score Matching: Treated vs. Matched Controls")
plt.legend()
plt.show()

"""
For each treated individual (someone who got the treatment), it finds a “similar” control individual 
(someone who didn’t get the treatment) with a similar propensity score. 
The propensity score is the probability of getting the treatment given their characteristics.

It calculates the difference in outcomes between the treated and their matched control.
It averages all these differences.
"""
matched_effect = (treated.outcome.values - matched_controls.outcome.values).mean()
print("Matching estimate of treatment effect:", matched_effect)

# IPTW (Inverse Probability of Treatment Weighting)
"""
Each person is given a weight based on how likely they were to get the treatment:
If treated: weight = 1 / propensity score
If control: weight = 1 / (1 − propensity score)
This gives more importance to people who are less likely to be in their group.
Then it calculates a weighted average outcome for treated and for control, and takes the difference.

Intuition:
“Let’s pretend our sample is perfectly balanced by weighting people so that treated and control groups look like they 
could have been randomized.”

Output:
weighted_effect is the average treatment effect adjusted for differences in baseline characteristics.
"""
df["weight"] = np.where(df.treatment == 1, 1/df["propensity"], 1/(1-df["propensity"]))
weighted_effect = np.average(df["outcome"] * df["treatment"], weights=df["weight"]) - \
                  np.average(df["outcome"] * (1-df["treatment"]), weights=df["weight"])
print("IPTW estimate of treatment effect:", weighted_effect)
