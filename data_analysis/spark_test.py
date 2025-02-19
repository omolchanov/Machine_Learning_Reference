from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
import requests

import pandas as pd

df = pd.read_csv('../assets/indian_flights.csv')


spark = SparkSession.builder.master("local[*]").appName("spark_on_docker").getOrCreate()

url = 'https://github.com/jupyter/docker-stacks/blob/d990a62010aededcda836196c4b04efece7f838f/'
filename = 'README.md '

# Prepare data
assembler = VectorAssembler(inputCols=["feature1", "feature2"], outputCol="features")
data = assembler.transform(df)
# Create and train model
lr = LogisticRegression(maxIter=10, regParam=0.01)
model = lr.fit(data)
# Make predictions
predictions = model.transform(data)
