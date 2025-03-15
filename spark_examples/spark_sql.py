# https://spark.apache.org/docs/latest/sql-getting-started.html

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('RemoteTest').remote('sc://localhost:15002').getOrCreate()

df = spark.read.json('../examples/src/main/resources/people.json')
# print(df.show())
# print(df.printSchema())
# print(df.select('age').show())
# print(df.filter(df.age < 21).show())


# Running SQL Queries Programmatically
df.createOrReplaceTempView('people')
result = spark.sql("SELECT * FROM people WHERE age > 21")
print(result.show())





