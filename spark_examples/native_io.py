from datetime import datetime, date

from pyspark.sql import SparkSession, Row

spark = SparkSession.builder.getOrCreate()
df = spark.read.csv('bank.csv', header=True, sep=';')

df.show()

# Write dataframe to CSV
df = spark.createDataFrame([
    Row(a=1, b=2., c='string1', d=date(2000, 1, 1), e=datetime(2000, 1, 1, 12, 0)),
    Row(a=2, b=3., c='string2', d=date(2000, 2, 1), e=datetime(2000, 1, 2, 12, 0)),
    Row(a=4, b=5., c='string3', d=date(2000, 3, 1), e=datetime(2000, 1, 3, 12, 0))
])

# df.write.csv('foo.csv', header=False, mode='overwrite')
