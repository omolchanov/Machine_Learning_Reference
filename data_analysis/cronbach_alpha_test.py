# https://www.statology.org/internal-consistency/
# https://www.statology.org/cronbachs-alpha-in-python/
# https://uedufy.com/how-to-calculate-cronbachs-alpha-in-excel/

import pandas as pd
import pingouin as pg

df = pd.DataFrame({
    'Q1': [1, 2, 2, 3, 2, 2, 3, 3, 2, 3],
    'Q2': [1, 1, 1, 2, 3, 3, 2, 3, 3, 3],
    'Q3': [1, 1, 2, 1, 2, 3, 3, 3, 2, 3]
})

'''
Cronbach’s Alpha ranges between 0 and 1, with higher values indicating that the survey or questionnaire
is more reliable.

Cronbach’s Alpha	Internal consistency
0.9 ≤ α	            Excellent
0.8 ≤ α < 0.9	    Good
0.7 ≤ α < 0.8	    Acceptable
0.6 ≤ α < 0.7	    Questionable
0.5 ≤ α < 0.6	    Poor
α < 0.5	            Unacceptable
'''

a = pg.cronbach_alpha(df)
print('Cronbach Alpha: ', a[0])
print('Confidence interval: ', a[1])
