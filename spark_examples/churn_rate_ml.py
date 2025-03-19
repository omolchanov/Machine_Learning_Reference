# https://www.statology.org/pyspark-train-test-split/
# https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/distributed-ml-for-spark-connect

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

import pandas as pd

import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import plot

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)

spark = SparkSession.builder.appName('RemoteTest').remote('sc://localhost:15002').getOrCreate()
df = spark.read.csv('../work-dir/csv/churn-rate.csv', header=True)

# Casting columns' types to the required types
columns_to_cast = [
    ('account_length', 'integer'),
    ('number_vmail_messages', 'integer'),
    ('total_day_minutes', 'float'),
    ('total_day_calls', 'integer'),
    ('total_day_charge', 'float'),
    ('total_eve_minutes', 'float'),
    ('total_eve_calls', 'integer'),
    ('total_eve_charge', 'float'),
    ('total_night_minutes', 'float'),
    ('total_night_calls', 'integer'),
    ('total_night_charge', 'float'),
    ('total_intl_minutes', 'float'),
    ('total_intl_calls', 'integer'),
    ('total_intl_charge', 'float'),
    ('customer_service_calls', 'integer')
]

for i,c in enumerate(columns_to_cast):
    df = df.withColumn(c[0], col(c[0]).cast(c[1]))

df.printSchema()
df.show()

# Dropping unnecessary columns
df = df.drop('area code', 'phone number')


def perform_descriptive_statistics():
    # Converting Spark Dataframe to Pandas Dataframe due to performing descriptive statistics
    pandas_df = df.toPandas()

    print('\nDescribing:', pandas_df.describe().round(3))
    print('\nDistribution of the target variable\n', pandas_df['churn'].value_counts())


def perform_eda():
    # Performing EDA and plotting
    df.createOrReplaceTempView('df')

    res_1 = spark.sql(
        "SELECT state, COUNT(churn) "
        "FROM df "
        "WHERE churn = 'True' "
        "GROUP BY state ORDER BY count(churn) DESC"
    ).toPandas()

    res_2 = spark.sql(
        "SELECT customer_service_calls, COUNT(account_length) "
        "FROM df "
        "WHERE churn = 'True' "
        "GROUP BY customer_service_calls "
        "ORDER BY customer_service_calls"
    ).toPandas()

    res_3 = spark.sql("SELECT state, churn, total_day_calls FROM df").toPandas()
    res_4 = spark.sql("SELECT state, churn, total_night_calls FROM df").toPandas()

    figures = [
        (px.bar(res_1, x='state', y='count(churn)'), (1,1)),
        (px.bar(res_2, x='customer_service_calls', y='count(account_length)'), (1,2)),

        (px.histogram(
            res_3,
            x='state',
            y='total_day_calls',
            color='churn',
            barmode='group',
            histfunc='count',
            height=400), (2,1)),

        (px.histogram(
            res_4,
            x='state',
            y='total_night_calls',
            color='churn',
            barmode='group',
            histfunc='count',
            height=400), (2,2))
    ]

    figures_names = [
        'Churned customers by state',
        'Churned customers by number of customer_service_calls',
        'Customers churn by state and total_day_calls',
        'Customers churn by state and total_night_calls'
    ]

    fig = make_subplots(rows=2, cols=2, subplot_titles=figures_names)

    for i, figure in enumerate(figures):
        for trace in range(len(figure[0]['data'])):
            fig.add_trace(figure[0]['data'][trace], row=figure[1][0], col=figure[1][1])

    fig.update_layout(showlegend=False)
    plot(fig)







