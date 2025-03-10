from datetime import datetime, date

from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import upper, pandas_udf

import pandas as pd

spark = SparkSession.builder.appName('RemoteTest').remote("sc://localhost:15002").create()

# Configuration
spark.conf.set('spark.sql.repl.eagerEval.maxNumRows', 12)


df = spark.createDataFrame([
    Row(a=1, b=2., c='string1', d=date(2000, 1, 1), e=datetime(2000, 1, 1, 12, 0)),
    Row(a=2, b=3., c='string2', d=date(2000, 2, 1), e=datetime(2000, 1, 2, 12, 0)),
    Row(a=4, b=5., c='string3', d=date(2000, 3, 1), e=datetime(2000, 1, 3, 12, 0))
])
print(df.head())

pandas_df = pd.read_csv('../assets/bank.csv', delimiter=';')
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


# Grouping data
df.groupby('education').avg().show()


# Register the DataFrame as a table and run a SQL
df.createOrReplaceTempView('df_main')
spark.sql("SELECT job FROM df_main WHERE age > 20").show()
spark.sql("SELECT * FROM df_main WHERE age IN (20, 30)").show(100)

spark.stop()
