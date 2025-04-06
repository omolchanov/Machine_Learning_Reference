import sys

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

from pyod.models.iforest import IForest

import pandas as pd
import numpy as np

import sweetviz as sv

import plotly.express as px
from plotly.offline import plot

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(threshold=sys.maxsize, suppress=True)


def prepare_df() -> pd.DataFrame:
    df = pd.read_csv('../assets/autos.csv')

    '''
    “symboling”, corresponds to a car’s insurance risk level. Cars are initially assigned a risk factor symbol that 
    corresponds to their price. If an automobile is more dangerous, this symbol is adjusted by increasing it. 
    A value of +3 indicates that the vehicle is risky, while -3 indicates that it is likely safe to insure.
    '''

    # Checking Nan and empty values
    # print('NaN values:\n', df.isna().sum())
    # print('Empty values:\n', df.isnull().sum())

    # print(df.info())

    cat_columns_to_label = df[[
        'make',
        'fuel_type',
        'aspiration',
        'body_style',
        'drive_wheels',
        'engine_location',
        'engine_type',
        'fuel_system'
    ]].columns
    num_columns_to_scale = df[[
        'wheel_base',
        'length',
        'width',
        'height',
        'curb_weight',
        'engine_size',
        'bore',
        'stroke',
        'horsepower',
        'peak_rpm',
        'peak_rpm',
        'highway_mpg',
        'price'
    ]].columns

    cat_pipeline = Pipeline(steps=[
        ('enc', OrdinalEncoder())
    ])

    num_pipeline = Pipeline(steps=[
        ('std', StandardScaler()),
        ('mm', MinMaxScaler())
    ])

    ct = ColumnTransformer([
        ('num_pipeline', num_pipeline, num_columns_to_scale),
        ('cat_pipeline', cat_pipeline, cat_columns_to_label)
    ])

    columns = num_columns_to_scale.tolist() + cat_columns_to_label.tolist()
    df[columns] = ct.fit_transform(df)

    df[cat_columns_to_label] = df[cat_columns_to_label].astype('int64')
    df['make'] = df['make'].astype('object')
    df['num_of_doors'] = df['num_of_doors'].astype('int64')

    # Dealing with outliers
    model = IForest(n_estimators=1)
    model.fit(df)

    labels = model.labels_
    df = df[labels == 0]

    print(df.shape)

    # Remove skewness from particular features
    def remove_skewness(feature, max_thr=1.0, min_thr=0.0):
        df_ns = df[(df[feature] > min_thr) & (df[feature] < max_thr)]
        return df_ns

    df = remove_skewness('wheel_base', 0.8, 0.2)
    df = remove_skewness('length', 0.8)
    df = remove_skewness('width', 0.8, 0.2)
    df = remove_skewness('height', 1, 0.2)
    df = remove_skewness('curb_weight', 0.7)
    df = remove_skewness('engine_size', 0.45, 0.1)
    df = remove_skewness('bore', 0.85, 0.25)
    df = remove_skewness('stroke', 1, 0.25)
    df = remove_skewness('peak_rpm', 0.7)
    df = remove_skewness('highway_mpg', 0.6)

    # Generate HTML report
    report = sv.analyze(df)
    report.show_html('ReportAutos.html')

    return df


def perform_clustering(X, cls) -> np.array:
    print(cls)
    print('X shape for clustering:', X.shape)

    cls.fit(X)

    labels = cls.labels_

    try:
        y_pred = cls.predict(X)
        print('Silhouette Score: ', silhouette_score(X, y_pred))
    except:
        y_pred = cls.fit_predict(X)
        print('Silhouette Score: ', silhouette_score(X, y_pred))

    try:
        print('Number of clusters: ', cls.n_clusters, '\n')
    except:
        print(cls.labels_)

    return labels


def plot_clusters(x, y, labels, title, x_axis, y_axis) -> None:
    fig = px.scatter(
        x=x.values,
        y=y.values,
        color=labels,
        title=title,
        labels={
            'x': x_axis,
            'y': y_axis,
            'color': 'cluster'
        }
    )

    plot(fig)


def prepare_clustering_principle(name: str, features: list) -> str | pd.DataFrame:
    df = prepare_df()
    X = df[features]

    return name, X


def cluster(principle, X, feat1, feat2):
    clss = [
        AffinityPropagation(verbose=True),
        AgglomerativeClustering(linkage='ward'),
        KMeans(n_clusters=4)
    ]

    for _, cls in enumerate(clss):
        labels = perform_clustering(X, cls)
        plot_clusters(
            x=X[feat1],
            y=X[feat2],
            labels=labels,
            title=str(cls) + ' ' + principle,
            x_axis=feat1,
            y_axis=feat2
        )


def run():
    principle1, X1 = prepare_clustering_principle('Clustering on segment', [
        'make',
        'body_style',
        'horsepower',
        'price'
    ])

    cluster(principle1, X1, 'make', 'price')

    principle2, X2 = prepare_clustering_principle('Insurance', [
        'symboling',
        'aspiration',
        'price',
        'num_of_cylinders'
    ])

    cluster(principle2, X2, 'symboling', 'price')


if __name__ == '__main__':
    run()
