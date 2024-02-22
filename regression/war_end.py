from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_absolute_error, r2_score


import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
pd.set_option('display.float_format', lambda x: '%.i' % x)

plt.rc('font', size=10)

# Source: https://www.kaggle.com/datasets/piterfm/2022-ukraine-russian-war
# Loading the datasets for equipment and personnel
df_eq = pd.read_csv('../assets/russia_losses_equipment.csv')
df_p = pd.read_csv('../assets/russia_losses_personnel.csv')

# Extracting and renaming the required columns
df_eq = df_eq[['tank', 'APC']]
df_eq.columns = ['tank_lost', 'APC_lost']

df_p = df_p[['personnel']]
df_p.columns = ['personnel_lost']

# Finding the delta of the equipment's and personell loses
df_eq = df_eq.diff().fillna(df_eq).astype(int)
df_p = df_p.diff().fillna(df_p).astype(int)

# Merging two datasets into one
df_f = pd.concat([df_eq, df_p], axis=1)


def find_correlation():
    """
    Finds the correlation between loses of personnel and tanks/APCs
    """

    print('Correlation Tanks <=> Personell: %.3f' % df_f['tank_lost'].corr(df_f['personnel_lost']))
    print('Correlation APC <=> Personell: %.3f' % df_f['APC_lost'].corr(df_f['personnel_lost']))


find_correlation()

# Splitting the data
X_f = df_f[['tank_lost', 'APC_lost']]
y_f = df_f['personnel_lost']

# Selecting a regressor
reg = LinearRegression()
reg.fit(X_f, y_f)

# Fitting the regressor and predicting
y_pred_f = reg.predict(X_f)
r2 = r2_score(y_f, y_pred_f)

print('R2 Score: %.3f' % r2)


def predict():
    """
    Returns the prediction basing on R2 score. The assumption is that personnel's loses is explained by loses of
    equipment (tanks, APCs)

    :return:
        - 1 if the model explains more than 75% of cases
        - 0 if the model explains less than 75% of cases
    """

    return 0 if r2 < 0.75 else 1


def evaluate_models():
    """
    Evaluates Linear, Ridge and Lasso regressors with MAE metric
    """

    X = df_f[['tank_lost', 'APC_lost']]
    y = df_f['personnel_lost']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    def evaluate_model(m):
        print('\n', m)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)

        if isinstance(m, Ridge) or isinstance(m, Lasso):
            print('Alpha: %.6f MAE: %.3f' % (m.alpha, mean_absolute_error(y_test, y_pred)))

        if isinstance(m, LinearRegression):
            print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

    # Ridge regression
    alpha_values = [0.00001, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 3, 5, 6, 7, 8, 9, 10]
    for a in alpha_values:
        evaluate_model(Ridge(alpha=a))

    # Lasso regression
    alpha_values = [0.000001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1]
    for a in alpha_values:
        evaluate_model(Lasso(alpha=a))

    # Linear Regression
    evaluate_model(LinearRegression())


def cross_validate_models():
    """
    Cross validates the models (Linear, Ridge, Lasso) with different number of folds
    in order to find if the model is overfitted or underfitted
    """

    X = df_f[['tank_lost', 'APC_lost']]
    y = df_f['personnel_lost']

    k = 12

    regs = [LinearRegression(), Lasso(), Ridge()]
    for reg in regs:
        folds = range(2, k + 1)
        test_means, train_means = [], []

        for f in folds:
            skf = KFold(n_splits=f)
            results = cross_validate(reg, X, y, cv=skf, return_train_score=True, scoring='neg_mean_absolute_error')

            test_scores = abs(results['test_score'])
            train_scores = abs(results['train_score'])

            test_means.append(np.mean(test_scores))
            train_means.append(np.mean(train_scores))

        plt.plot(folds, test_means)
        plt.plot(folds, train_means)

        plt.title(['Cross Validation', reg])
        plt.legend(['Test scores', 'Train scores'])
        plt.xlabel('Fold')
        plt.ylabel('Mean MAE')

        plt.show()


pred = predict()
print('Prediction: %i' % pred)

evaluate_models()
cross_validate_models()
