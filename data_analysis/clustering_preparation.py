# Guidelines:
# https://www.displayr.com/understanding-cluster-analysis-a-comprehensive-guide/
# https://medium.com/@evgen.ryzhkov/5-stages-of-data-preprocessing-for-k-means-clustering-b755426f9932
# https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/

import sys

from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from sklearn.manifold import TSNE

import scipy as sp
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import Normalizer

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
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv('../assets/winequality-white.csv', sep=';')
df = df.iloc[:, :-1]


def build_wiskey_plot(df_o):
    fig, ax = plt.subplots(1, df_o.shape[1], sharex=False)

    for i, c in enumerate(df_o.columns):
        sns.boxplot(y=df_o[c], ax=ax[i])

    plt.subplots_adjust(wspace=1.75)
    plt.show()


def build_distribution_plots(df_o):
    fig, ax = plt.subplots(1, df_o.shape[1])

    for i, c in enumerate(df_o.columns):
        sns.kdeplot(df_o[c], ax=ax[i]).set(xlabel='', ylabel='')

    plt.subplots_adjust(wspace=0.5)
    plt.show()


def build_clustering_plot(model, X, y_pred):
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_pred).set(title='Clusters: ' + str(model))
    plt.show()


def calculate_multicollinearity(df_o):
    """
    Multicollinearity detection
    If VIF 1-5 - ok, there is no strong correlation
    If VIF more than 5 - there is strong correlation
    If VIF is inf - there is extremely high correlation (linear dependence)
    :param df_o:
    :return:
    """
    vif = [variance_inflation_factor(df_o.values, i) for i in range(df_o.shape[1])]
    result = pd.DataFrame({'Variance Inflation Factor': vif[0:]}, index=df_o.columns).T

    print('\n', result)


def drop_duplicates(df_o):
    df_o.drop_duplicates(subset=None, keep='first', inplace=True)
    df_o.reset_index(drop=True, inplace=True)

    return df_o


def drop_outliers(df_o):
    """
    Removes the outliers from all the features basing InterQuartile Range
    """

    outlier_columns = df_o[[
        'fixed acidity',
        'volatile acidity',
        'chlorides',
        'sulphates',
        'citric acid',
        'pH',
        'free sulfur dioxide'
    ]]

    for c in outlier_columns:
        q1 = np.percentile(df_o[c], 25)
        q3 = np.percentile(df_o[c], 75)

        iqr = q3 - q1

        upper = np.where(df_o[c] >= (q3 + 1.4 * iqr))
        lower = np.where(df_o[c] <= (q3 - 1.5 * iqr))

        df_o.drop(upper[0], inplace=True)
        df_o.drop(lower[0], inplace=True)

        df_o.reset_index(drop=True, inplace=True)

    return df_o


def standardize_features(df_o):
    """
    Standardization is feature scaling method where the values are centered around the mean with a
    unit standard deviation. This means that the mean of the attribute becomes zero, and the resultant
    distribution has a unit standard deviation.
    :param df_o:
    :return:
    """

    skewed_values = df_o.apply(lambda x: sp.stats.skew(x)).sort_values(ascending=False)
    skewed_features = skewed_values[abs(skewed_values) > 0.5]

    for f in skewed_features.index:
        df_o[f] = sp.special.boxcox1p(df_o[f], sp.stats.boxcox_normmax(df_o[f] + 1))

    return df_o


def normalize_features(df_o):
    """
    Normalization, a vital aspect of Feature Scaling, is a data preprocessing technique employed to
    standardize the values of features in a dataset, bringing them to a common scale.
    :param df_o:
    :return:
    """

    scaled_data = Normalizer().fit_transform(df_o.values)
    df_t = pd.DataFrame(scaled_data, index=df_o.index, columns=df_o.columns)

    return df_t


def reduce_features(df_o):
    return TSNE(n_components=2).fit_transform(df_o)


def prepare_data():
    drop_duplicates(df)
    drop_outliers(df)

    X = standardize_features(df)
    X = normalize_features(X)

    drop_outliers(X)

    X = reduce_features(X)

    return X


def evaluate_model(X, model, plot_clusters=False):
    print('\n', model)

    try:
        model.fit(X)
        y_pred = model.predict(X)

    except:
        y_pred = model.fit_predict(X)

    print('Silhouette Score: %.3f' % silhouette_score(X, y_pred))

    # The Davies-Bouldin index calculates the average dissimilarity between each cluster
    # and its most similar cluster, with lower values indicating better clustering.
    print('Davies-Bouldin Index: %.3f' % davies_bouldin_score(X, y_pred))

    # The Calinski-Harabasz index quantifies the ratio of between-cluster dispersion to
    # within-cluster dispersion. Higher values suggest well-separated and compact clusters.
    print('Calinski-Harabasz Index: %.3f' % calinski_harabasz_score(X, y_pred))

    if plot_clusters is True:
        build_clustering_plot(model, X, y_pred)


if __name__ == '__main__':
    data = prepare_data()
    print(data.shape)

    models = [KMeans(n_clusters=7), AffinityPropagation(), AgglomerativeClustering(n_clusters=7)]

    for m in models:
        evaluate_model(data, m, plot_clusters=True)
