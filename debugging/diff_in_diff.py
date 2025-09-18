"""
A causal inference method used to estimate treatment effects (like a new policy, product change, or ad campaign).
It compares the before–after change in outcomes for a treatment group against the before–after change for a control
group.
DiD = (Ytreat,after − Ytreat,before)−(Ycontrol,after − Ycontrol,before)
"""

import pandas as pd
import statsmodels.api as sm

# Sample data: group (0=control, 1=treatment), time (0=before, 1=after), outcome
df = pd.DataFrame({
    "group": [0,0,0,0,1,1,1,1],
    "time":  [0,1,0,1,0,1,0,1],
    "outcome": [5,6,4,5,6,10,5,9]
})

# Difference-in-difference regression
df["treat_time"] = df["group"] * df["time"]
model = sm.OLS(df["outcome"], sm.add_constant(df[["group","time","treat_time"]]))

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                outcome   R-squared:                       0.937
Model:                            OLS   Adj. R-squared:                  0.889
Method:                 Least Squares   F-statistic:                     19.67
Date:                Thu, 18 Sep 2025   Prob (F-statistic):            0.00740
Time:                        23:55:10   Log-Likelihood:                -5.8063
No. Observations:                   8   AIC:                             19.61
Df Residuals:                       4   BIC:                             19.93
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          4.5000      0.500      9.000      0.001       3.112       5.888
group          1.0000      0.707      1.414      0.230      -0.963       2.963
time           1.0000      0.707      1.414      0.230      -0.963       2.963
treat_time     3.0000      1.000      3.000      0.040       0.224       5.776
==============================================================================
Omnibus:                        9.677   Durbin-Watson:                   1.500
Prob(Omnibus):                  0.008   Jarque-Bera (JB):                1.333
Skew:                           0.000   Prob(JB):                        0.513
Kurtosis:                       1.000   Cond. No.                         6.85
==============================================================================

const (β₀ = 4.5)
This is the baseline outcome.
It represents the control group, before treatment → average outcome = 4.5. Matches your data table.

group (β₁ = 1.0, p=0.23)
Difference between treatment and control groups before treatment.
Treatment group started 1 unit higher than control (not statistically significant → p=0.23).
So before treatment, groups were roughly similar.

time (β₂ = 1.0, p=0.23)
Change over time in the control group.
Control outcome increased by 1 unit from before → after (not significant, but small sample size).

treat_time (β₃ = 3.0, p=0.04)
This is the DiD effect = treatment effect.
Interpretation: After treatment, the treatment group improved by 3 units more than the control group did.
Statistically significant at 5% (p=0.04).

R² and Model Fit
R-squared = 0.937 → model explains ~94% of outcome variance.
Good fit (but keep in mind: only 8 observations → small sample).
"""

result = model.fit()
print(result.summary())
