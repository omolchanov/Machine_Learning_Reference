# https://medium.com/@techwithpraisejames/hypothesis-testing-with-python-t-test-z-test-and-p-values-code-examples-fa274dc58c36
# https://www.datacamp.com/tutorial/an-introduction-to-python-t-tests
# https://www.datacamp.com/tutorial/t-test-vs-z-test

import random

import numpy as np
from scipy import stats


def one_sample_ttest():
    """
    Higher values of the t-score indicate that a large difference exists between the two sample sets. The smaller the
    t-value, the more similarity exists between the two sample sets. A large t-score, or t-value, indicates that the
    groups are different while a small t-score indicates that the groups are similar.

    If the p-value is less than the significance level, then H₁ is true, so we reject H₀. If the p-value is greater
    than the significance level, then H₀ is true, so we accept H₀

    H0 hypothesis: The sample mean = population mean(45)
    H1 hypothesis: The sample mean ≠ population mean(45)

    Interpretation of the results:
      -- The p-value is less than 0.05(significance level)
      -- The sample mean(appr. 53) is also ≠ population mean(45)
    The H0 hypothesis is rejected
    """

    a = [random.gauss(50, 20) for x in range(30)]

    t_stat, p_value = stats.ttest_1samp(a, 45)

    print('T-statistics: ', t_stat)
    print('p-value: ', p_value)
    print('Sample mean: ', np.mean(a))


def two_sample_ttest():
    """
    Two-sample or Independent t-test example

    H0 hypothesis: mean of sample A = mean of sample B
    H1 hypothesis: mean of sample A ≠ mean of sample B

    Interpretation of the results: The p-value is greater than 0.05(significance level) and the mean of a = mean of b
    The H0 hypothesis is accepted
    """

    a = [random.gauss(50, 20) for x in range(30)]
    b = [random.gauss(55, 15) for x in range(30)]

    t_stat, p_value = stats.ttest_ind(a, b)

    print('\nT-statistics: ', t_stat)
    print('p-value: ', p_value)
    print('Sample A mean: ', np.mean(a))
    print('Sample B mean: ', np.mean(b))


def paired_ttest():
    """
    Paired t-test example
    H0 hypothesis: mean of sample B - mean of sample A = 0
    H1 hypothesis: mean of sample B - mean of sample A ≠ 0

    Interpretation of the results: The p-value is less than 0.05(significance level) and the mean of b - mean of a ≠ 0
    The H0 hypothesis is rejected
    """

    a = [random.gauss(50, 20) for x in range(30)]
    b = [random.gauss(60, 25) for x in range(30)]

    t_stat, p_value = stats.ttest_rel(a, b)

    print('\nT-statistics: ', t_stat)
    print('p-value: ', p_value)

    print('Min difference: ', np.mean(a) - np.mean(b))


def welch_ttest():
    """
    Welch's t-test example. Unlike other t-tests, Welch's t-test is applicable for unequal population variances
    H0 hypothesis: mean of sample B - mean of sample A = 0
    H1 hypothesis: mean of sample B - mean of sample A ≠ 0

    Interpretation of the results: The p-value is greater than 0.05(significance level)
    The H0 hypothesis is accepted
    """

    a = [random.gauss(50, 20) for x in range(30)]
    b = [random.gauss(60, 25) for x in range(50)]

    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

    print('\nT-statistics: ', t_stat)
    print('p-value: ', p_value)

    print('Min difference: ', np.mean(a) - np.mean(b))


one_sample_ttest()
two_sample_ttest()
paired_ttest()
welch_ttest()
