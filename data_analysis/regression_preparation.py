# Guidelines:
# https://towardsdatascience.com/regression-data-case-study-e45d915c8cf2
# https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/assumptions-of-multiple-linear-regression/
# https://www.youtube.com/watch?v=rw84t7QU2O0

import sys

import scipy as sp

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, cross_validate, train_test_split

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.outliers_influence import variance_inflation_factor

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
np.set_printoptions(threshold=np.inf, suppress=True)
np.set_printoptions(threshold=sys.maxsize, suppress=True)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)


df = pd.read_csv('../assets/petrol_consumption.csv')

X = df.iloc[:, :-1]
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the OLS model, make preidctions, calculate residuals
X_with_constant = sm.add_constant(X_train)
model = sm.OLS(y_train, X_with_constant)

results = model.fit()

X_test = sm.add_constant(X_test)
y_pred = results.predict(X_test)

residulas = y_test - y_pred


def print_df_summary():
    print('\n Shape', df.shape)

    print('\n The summary stats: ')
    print(df.describe())

    print('\n First 5 rows: \n')
    print(df.head(5))


def print_ols_model_summary():
    print('\n', results.summary())


def check_features_correlation():
    sns.heatmap(df.corr(method='pearson'), annot=True).set(
        title='Correlation between the features'
    )
    plt.show()


def check_multicolinearity():
    """
    https://www.investopedia.com/terms/v/variance-inflation-factor.asp

    Multicollinearity exists when there is a correlation between multiple independent variables in a
    multiple regression model. This can adversely affect the regression results.

    VIF value > 10 signifes the heavy multicollinearity
    VIF value < 5 signiges the little relationship between features
    """

    vif = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    result = pd.DataFrame({'Variance Inflation Factor': vif[0:]}, index=X_train.columns).T

    print('\n', result)


def check_multivariative_normality():
    """
    Checks if  the residuals are normally distributed.

    The residuals are centered around 0  what is assumption of normal distibution
    If theoretical and observed values fall almost on the same line on P-P plot, that the overall
    distribution is normal
    The mean value of distribution should be near 0 for normally distributed residuals
    """

    print('\n Residuals: \n', residulas)

    sns.kdeplot(residulas).set(title='Distribution of Residuals')
    plt.show()

    sp.stats.probplot(residulas, fit=True, plot=plt)
    plt.title('Residuals P-P plot')
    plt.show()

    print('Residuals` mean: ', np.mean(residulas))


def check_homoscedasticity():
    """
    Homoscedasticity, or homogeneity of variances, is an assumption of equal or similar variances in
    different groups being compared.
    This is an important assumption of parametric statistical tests because they are sensitive to
    any dissimilarities. Uneven variances in samples result in biased and skewed test results.

    If the values are centered arounf 0 and there are no patterns of grouping, the assumption of the
    absence of the Homoscedasticity is true.
    """

    sns.scatterplot(x=y_pred, y=residulas).set(
        title='Homogeneity of variances',
        xlabel='y_pred',
        ylabel='residuals'
    )
    plt.show()


def check_no_autocorrelation():
    """
    Show how residuals are correlated with themselves. If the threshold line is crossed, then the value
    has a significant autocorrelation.
    There should be no significant autocorrelation between the resilduals
    """

    plot_acf(residulas, lags=9)

    plt.title('Autocorrelation plot')
    plt.xlabel('Residuals')
    plt.ylabel('Correlation')

    plt.show()


def check_distribution():
    """
    Check if the variables (features and target variable) are normally distributed
    """

    for c in df.columns:
        sns.kdeplot(df[c].values).set(title='Distribution of ' + c)
        plt.show()


def fix_skewness(plot_fixed_features=False):
    """
    Apart from assumptions, the modeling step also includes the data to follow the statistical concepts,
    for example, normal distribution. Our data must be normalized. The skewness of data is fixed using scipy
    statistical module the function.

    Skewness refers to distortion or asymmetry in a symmetrical bell curve, or normal distribution,
    in a set of data. If the curve is shifted to the left or to the right, it is said to be skewed.
    """

    # Identifying the skewed features
    numeric = X.dtypes[df.dtypes != 'object'].index

    skewed_values = df[numeric].apply(lambda x: sp.stats.skew(x)).sort_values(ascending=False)
    skewed_features = skewed_values[abs(skewed_values) > 0.5]

    print('\n Skewed features: \n', skewed_features)

    # Fixing the skewness of the feature with Box-Cox transformation
    # https://builtin.com/data-science/box-cox-transformation-target-variable
    for f in skewed_features.index:
        X[f] = sp.special.boxcox1p(X[f], sp.stats.boxcox_normmax(X[f] + 1))

    # Plotting the distribution for the fixed features
    if plot_fixed_features is True:
        for ff in skewed_features.index:
            sns.kdeplot(X[ff].values).set(title='Fixed ' + ff)
            plt.show()


def cross_validation(model):

    k = 5
    cv = KFold(n_splits=k)

    scores = cross_validate(model, X, y, cv=cv, scoring=['r2', 'neg_mean_absolute_error'])
    return scores


if __name__ == '__main__':
    print_df_summary()
    print_ols_model_summary()
    check_features_correlation()

    check_multicolinearity()
    check_multivariative_normality()
    check_homoscedasticity()
    check_no_autocorrelation()

    check_distribution()
    fix_skewness(plot_fixed_features=True)

    models = (LinearRegression(), Lasso(), Ridge())

    for m in models:
        print('\n', m)
        print(X.shape)

        mean_r2 = np.mean(cross_validation(m)['test_r2'])
        mean_loss = np.mean(cross_validation(m)['test_neg_mean_absolute_error'])

        print('Mean R2: %.3f' % mean_r2)
        print('Mean loss: %.3f' % np.abs(mean_loss))
