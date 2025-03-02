from datetime import datetime, date

from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import upper, pandas_udf

import pandas as pd

spark = SparkSession.builder.getOrCreate()

# Configuration
spark.conf.set('spark.sql.repl.eagerEval.enabled', True)
spark.conf.set('spark.sql.repl.eagerEval.maxNumRows', 12)

df = spark.createDataFrame([
    Row(a=1, b=2., c='string1', d=date(2000, 1, 1), e=datetime(2000, 1, 1, 12, 0)),
    Row(a=2, b=3., c='string2', d=date(2000, 2, 1), e=datetime(2000, 1, 2, 12, 0)),
    Row(a=4, b=5., c='string3', d=date(2000, 3, 1), e=datetime(2000, 1, 3, 12, 0))
])
print(df)
print(df.head())

pandas_df = pd.read_csv('bank.csv', delimiter=';')
df = spark.createDataFrame(pandas_df)

# Printing the data
print(df.printSchema())
print(df.show())
print(df.show(5, vertical=True))

# Printing columns and describing the data
print(df.columns)
print(df.select('age', 'balance', 'y').describe().show())

# DataFrame.collect() collects the distributed data to the driver side as the local data in Python
df.collect()
df.take(5)

# Converting Spark Dataframe to Pandas
df.toPandas()

# Getting a column
print(df.age)

# Selecting values of a column
df.select(df.education).show()

# Adding a new column to the dataframe
df = df.withColumn('!divided balance', df.balance / 2)
df = df.withColumn('!marital', upper(df.marital))
df.show()

# Filtering rows by criteria
df.filter(df.age < 30).filter(df.job == 'student').filter(df.loan == 'yes').show()


# Applying functions to the dataframe
# A pandas user-defined function (UDF)—also known as vectorized UDF—is a user-defined function that uses Apache Arrow
# to transfer data and pandas to work with the data.
@pandas_udf('string')
def pandas_plus_substring(series):
    return series + '_new_string'


df.select(pandas_plus_substring(df.education)).show()


# Using Pandas API
def pandas_filter_func(iterator):
    for pandas_df in iterator:
        yield pandas_df[pandas_df.age == 30]


df.mapInPandas(pandas_filter_func, schema=df.schema).show()

# Grouping data
df.groupby('education').avg().show()

# Register the DataFrame as a table and run a SQL
df.createOrReplaceTempView('df_main')
spark.sql("SELECT job FROM df_main WHERE age > 20").show()
spark.sql("SELECT * FROM df_main WHERE age IN (20, 30)").show(100)
