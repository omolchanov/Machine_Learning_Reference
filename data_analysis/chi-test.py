# https://www.stratascratch.com/blog/chi-square-test-in-python-a-technical-guide/
# https://www.statology.org/chi-square-critical-value-python/
# https://thedatascientist.com/high-p-value-and-low-chi-squared-in-stata/

from scipy.stats import chi2_contingency, chi2

import pandas as pd
import numpy as np

# Configuration
np.set_printoptions(threshold=np.inf, suppress=True)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)

# Dataset description
'''
1 school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
2 sex - student's sex (binary: 'F' - female or 'M' - male)
3 age - student's age (numeric: from 15 to 22)
4 address - student's home address type (binary: 'U' - urban or 'R' - rural)
5 famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
6 Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
7 Medu - mother's education (numeric: 
    0 - none, 
    1 - primary education (4th grade), 
    2 - 5th to 9th grade, 
    3 -secondary education, 
    4 - higher education)
8 Fedu - father's education (numeric: 
    0 - none, 
    1 - primary education (4th grade), 
    2 - 5th to 9th grade, 
    3 - secondary education, 
    4 - higher education)
9 Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services', 'at_home' or 'other')
10 Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services', 'at_home' or 'other')
11 reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
12 guardian - student's guardian (nominal: 'mother', 'father' or 'other')
13 traveltime - home to school travel time (numeric: 
    1 - <15 min., 
    2 - 15 to 30 min., 
    3 - 30 min. to 1 hour, 4 - >1 hour)
14 studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
15 failures - number of past class failures (numeric: n if 1<=n<3, else 4)
16 schoolsup - extra educational support (binary: yes or no)
17 famsup - family educational support (binary: yes or no)
18 paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
19 activities - extra-curricular activities (binary: yes or no)
20 nursery - attended nursery school (binary: yes or no)
21 higher - wants to take higher education (binary: yes or no)
22 internet - Internet access at home (binary: yes or no)
23 romantic - with a romantic relationship (binary: yes or no)
24 famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
25 freetime - free time after school (numeric: from 1 - very low to 5 - very high)
26 goout - going out with friends (numeric: from 1 - very low to 5 - very high)
27 Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
28 Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
29 health - current health status (numeric: from 1 - very bad to 5 - very good)
30 absences - number of school absences (numeric: from 0 to 93)

31 G1 - first period grade (numeric: from 0 to 20)
31 G2 - second period grade (numeric: from 0 to 20)
32 G3 - final grade (numeric: from 0 to 20, output target)
'''

df = pd.read_csv('../assets/student-mat.csv', sep=';')

# print(df.info())
# print(df.head())
# print(df.describe())


def perform_chi_squad_test(feature1, feature2):
    """
    A function to perform the Chi-Square Test and interpret results.
    Significance level = 0.05

    If a chi-square calculated value is greater than the chi-square critical value,
    then you reject your null hypothesis.

    Degrees of freedom for the chi-square using the following formula: df = (r-1)(c-1), where r is the number of rows
    and c is the number of columns. If the observed chi-square test statistic exceeds the critical value,
    you can reject the null hypothesis.

    To calculate Expected values (E) multiply each row total by each column total and divide by the overall total.
    """
    a = 0.05

    contingency_table = pd.crosstab(df[feature1], df[feature2])
    chi2_value, p, dof, expected = chi2_contingency(contingency_table)

    print(contingency_table)
    print(expected)

    # Calculating the chi-square critical value
    cv = chi2.ppf(1-a, dof)

    # Interpreting the result
    # A large chi-squared value, combined with a small p-value, suggests a significant relationship between the
    # variables being analyzed. Conversely, a small chi-squared value and a large p-value indicate weak or
    # no evidence of association.
    return chi2_value, p, p < a and chi2_value > cv, cv


aspects_and_features = {
    'School Support and Academic Performance': ('schoolsup', 'G3'),
    'Family Support and Grades': ('famsup', 'G3'),
    'Extra-Curricular Activities and Performance': ('activities', 'G3'),
    'Romantic Relationships and Academic Performance': ('romantic', 'G3'),
    'Health Status and Grades': ('health', 'G3')
}

for a, f in enumerate(aspects_and_features):
    feature1 = aspects_and_features[f][0]
    feature2 = aspects_and_features[f][1]

    print('\n', f)

    result = perform_chi_squad_test(feature1, feature2)
    print('\nChi2: %.3f | Critical chi2 value: %.3f | p-value %.3f | Is valuable: %s' %
          (result[0], result[3], result[1], result[2]))
