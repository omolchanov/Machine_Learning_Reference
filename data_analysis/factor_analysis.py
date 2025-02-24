# https://www.datacamp.com/tutorial/introduction-factor-analysis

from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from factor_analyzer import FactorAnalyzer

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Configuration
np.set_printoptions(threshold=np.inf, suppress=True)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)

df = pd.read_csv('../assets/bfi.csv')

print(df.info())

# Data preparation
df.drop(['rownames', 'gender', 'education', 'age'], axis=1, inplace=True)
df.dropna(inplace=True)

# Perform Adequacy test
# Evaluate the “factorability” of our dataset. Factorability means "can we found the factors in the dataset?"
# Bartlett’s test of sphericity checks whether the observed variables intercorrelate at all using the
# observed correlation matrix against the identity matrix. If the test found statistically insignificant,
# you should not employ a factor analysis.
chi_square, p_value = calculate_bartlett_sphericity(df)
print('Chi-square: %.3f' % chi_square)
print('p-value: %.3f' % p_value)

# Kaiser-Meyer-Olkin (KMO) Test measures the suitability of data for factor analysis. It determines the adequacy for
# each observed variable and for the complete model. KMO estimates the proportion of variance among all the observed
# variable. Lower proportion id more suitable for factor analysis. KMO values range between 0 and 1.
# Value of KMO less than 0.6 is considered inadequate.
kmo_all, kmo_model = calculate_kmo(df)
print('KMO model: %.3f' % kmo_model)

# Choosing the Number of Factors
# Kaiser criterion is used and scree plot. Both are based on eigenvalues.
fa = FactorAnalyzer()
fa.fit(df)
eigenvalues, vectors = fa.get_eigenvalues()

# Only for 6-factors eigenvalues are greater than 1. It means we need to choose only 6 factors
# (or unobserved variables).
print('\nEigenvalues', eigenvalues)
print('\nVectors: ', vectors)

# Create scree plot using matplotlib
plt.scatter(range(1, df.shape[1]+1), eigenvalues)
plt.plot(range(1, df.shape[1]+1), eigenvalues)
plt.axhline(1, 0, c='r')
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

# Performing Factor Analysis
fa.n_factors = 5
fa.rotation = 'varimax'
fa.fit(df)

factor_columns = ['Factor ' + str(i+1) for i in range(fa.n_factors)]

# Composing a dataframe with factors
df_factors = pd.DataFrame(fa.loadings_).round(3).abs()
df_factors.columns = factor_columns
df_factors['Feature'] = df.columns
print('\n', df_factors)

# Sorting each factor by the loading value, eliciting factors' features
for i, c in enumerate(df_factors.columns[:-1]):
    print(df_factors.sort_values(by=[c], ascending=False)[['Feature', c]].head(10))
    print('\n#########################')

# Get variance of each factors
# Total 42% cumulative Variance explained by the 5 factors.
df_variance = pd.DataFrame(fa.get_factor_variance())
df_variance.columns = factor_columns
df_variance.insert(0, 'metrics', '')
df_variance['metrics'] = ['SS Loadings', 'Proportion Var', 'Cumulative Var']
print('\n', df_variance)
