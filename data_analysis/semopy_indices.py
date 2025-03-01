# https://semopy.com/tutorial.html
# https://en.wikipedia.org/wiki/Confirmatory_factor_analysis

import semopy
import pandas as pd

df = pd.read_csv('../assets/boston_houses.csv', delim_whitespace=True)

# x: average number of rooms per dwelling
# y: Median value of owner-occupied homes in $1000's
x = df.iloc[:, 5]
y = df.iloc[:, -1]

data = pd.DataFrame({'x': x, 'y': y})

m = semopy.examples.univariate_regression.get_model()
model = semopy.Model(m)
model.fit(data)

print(model.inspect())

'''
The comparative fit index (CFI) analyzes the model fit by examining the discrepancy between the data and the 
hypothesized model, while adjusting for the issues of sample size inherent in the chi-squared test of model fit, 
and the normed fit index. 
CFI values range from 0 to 1, with larger values indicating better fit. CFI value of .95 or higher is presently accepted 
as an indicator of good fit.

The normed fit index (NFI) analyzes the discrepancy between the chi-squared value of the hypothesized model and the 
chi-squared value of the null model. The non-normed fit index (NNFI; also known as the Tucker–Lewis index) resolves some 
of the issues of negative bias, though NNFI values may sometimes fall beyond the 0 to 1 range. 
Values for both the NFI and NNFI should range between 0 and 1, with a cutoff of .95 or greater indicating a 
good model fit.
'''
stats = semopy.calc_stats(model)
print(stats.T)
