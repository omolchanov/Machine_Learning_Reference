# https://spark.apache.org/docs/3.5.4/api/python/getting_started/quickstart_ps.html
# https://stackoverflow.com/questions/70988705/spark-dataframe-vs-pandas-on-spark-dataframe
# https://spark.apache.org/docs/3.5.1/sql-data-sources-orc.html
# https://spark.apache.org/docs/3.5.1/sql-data-sources-parquet.html


import warnings
warnings.filterwarnings('ignore')

from pyspark.sql import SparkSession
import pyspark.pandas as ps

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Configuration
np.set_printoptions(threshold=np.inf, suppress=True)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)


spark = SparkSession.builder.appName('RemoteTest').remote('sc://localhost:15002').getOrCreate()

series = ps.Series([1, 3, 5, np.nan, 6, 8])
print(series)

df = ps.DataFrame({
        'a': [1, 2, 3, 4, 5, 6],
        'b': [100, 200, 300, 400, 500, 600],
        'c': ["one", "two", "three", "four", "five", "six"]
    },
    index=[10, 20, 30, 40, 50, 60])
print(df)

# Creating a pandas DataFrame by passing a numpy array, with a datetime index and labeled columns
dates = pd.date_range('20250401', periods=6, freq='M')
pddf = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['A', 'B', 'C', 'D'])
psdf = ps.from_pandas(pddf)
print(psdf)

# Creating a Spark DataFrame from pandas DataFrame
sdf = spark.createDataFrame(pddf)
print(sdf.show())

# Creating pandas-on-Spark API DataFrame
psdf_api = sdf.pandas_api()
print(psdf_api)
print(psdf_api.index)
print(psdf_api.columns)
print(psdf_api.to_numpy())

# Quick statistic summary of the data
print(psdf_api.describe())

# Transpose
print(psdf_api.T)

# Sorting
print(psdf_api.sort_index(ascending=False))
print(psdf_api.sort_values(by='D', ascending=False))

pd_df1 = pddf.reindex(index=dates[0:4], columns=list(pddf.columns) + ['E'])
pd_df1.loc[dates[0]:dates[1], 'E'] = 1
print(pd_df1)

psdf_api1 = spark.createDataFrame(pd_df1).pandas_api()
psdf_api1 = psdf_api1.fillna(value=5)
psdf_api1 = psdf_api1.dropna(how='any')
print(psdf_api1)

# Performing a descriptive statistic
print(psdf_api1.mean())

# Grouping
psdf_1 = pd.DataFrame({
    'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
    'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
    'C': np.random.randn(8),
    'D': np.random.randn(8)
})

psdf_api2 = spark.createDataFrame(psdf_1).pandas_api()
print(psdf_api2.groupby('A').count())
print(psdf_api2.groupby('D').sum())
print(psdf_api2.groupby(['A', 'B']).count())

# Plotting
pser = spark.createDataFrame(
    pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
).pandas_api()
plt.plot(pser.index.values, pser.values)
plt.show()

# Reading CSV data
csv_df = ps.read_csv('../work-dir/csv/bank.csv', sep=';').to_pandas()
csv_spark_df = spark.createDataFrame(csv_df).pandas_api()
print(csv_spark_df.head(10))

# Writing and Reading Parquet file
# csv_spark_df.to_parquet('../work-dir/parquet/bank.parquet')
parquet_df = ps.read_parquet('../work-dir/parquet/bank.parquet')
parquet_spark_df = spark.createDataFrame(parquet_df).pandas_api()
print(parquet_df.head(10))

# Writing and Reading ORC file
# csv_spark_df.to_orc('../work-dir/orc/bank.orc')
orc_df = ps.read_orc('../work-dir/orc/bank.orc').to_pandas()
orc_spark_df = spark.createDataFrame(orc_df).pandas_api()
print(orc_spark_df.head(10))

spark.stop()
