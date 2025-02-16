# https://medium.com/@techwithpraisejames/hypothesis-testing-with-python-t-test-z-test-and-p-values-code-examples-fa274dc58c36

import random

import numpy as np
from statsmodels.stats.weightstats import ztest as ztest

random.seed(20)


def one_sample_ztest():
    """
    One Sample Z-test for sample 'a'

    H0 hypothesis: mean of a = 100
    H1 hypothesis: mean of a ≠ 100

    Interpretation of the results: The p-value is greater than 0.05, and the mean of A is close to 100
    The H0 hypothesis is accepted
    """

    # We will use random data for the sample where 100 is the mean(specified value) and 15 is the standard deviation
    a = [random.gauss(100, 15) for x in range(40)]
    z_stat, p_value = ztest(a, value=100)

    print('Z-statistics: ', z_stat)
    print('P-value: ', p_value)

    print('Sample A mean: ', np.mean(a))


def two_samples_ztest():
    """
    Two Sample Z-test for samples 'a' and 'b'

    H0 hypothesis: Mean of sample A = mean of sample B
    H1 hypothesis: Mean of sample A ≠ mean of sample B

    Interpretation of the results: the p-value is less than 0.05 and mean of a ≠ mean of b
    The H0 hypothesis is rejected
    """
    a = [random.gauss(100, 15) for x in range(40)]
    b = [random.gauss(120, 15) for x in range(40)]

    z_stat, p_value = ztest(a, b, value=0)

    print('Z-statistics: ', z_stat)
    print('P-value: %.10f' % p_value)

    print('Sample A mean: ', np.mean(a))
    print('Sample B mean: ', np.mean(b))


one_sample_ztest()
two_samples_ztest()
