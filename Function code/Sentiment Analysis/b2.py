from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import dump_svmlight_file
import numpy as np
import os
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pandas import Series,DataFrame

sc = SparkContext('local')
spark = SparkSession(sc)

# Load training data
data = spark.read.format("libsvm") \
    .load("/home/yk/yelp4/review4.json")

# Split the data into train and test
splits = data.randomSplit([0.9, 0.1], 12)
train = splits[0]
test = splits[1]


nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
model = nb.fit(train)
predictions = model.transform(test)
predictions.show()
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))


print(".........")
test2= spark.read.format("libsvm") \
    .load("/home/yk/yelp4/review3.json")

predictions2 = model.transform(test2)

print(type(predictions2))

# Select example rows to display.
result2=predictions2.select("prediction")
print(type(result2))

df2= result2.toPandas()
print(type(df2))
print(df2)


yy=df2.prediction
print(yy)

 
import json
with open("./review2.json") as data_file:
    data = json.load(data_file)

print(len(data));

import pandas as pd
raw_docs = []
urls=[]
labels = []
bids= []
rids= []
#sen_dic={"positive":1,"negative":-1,"neutral":0}
LL=range(len(data))
for i in range(len(data)):
    raw_docs.append(data[i]['text'])
    urls.append(data[i]['user_id'])
    labels.append(data[i]['stars'])
    bids.append(data[i]['business_id'])
    rids.append(data[i]['review_id'])

#print (raw_docs[0])

bids_series=pd.Series(bids)
star_series=pd.Series(labels)
zz=(yy*5+star_series)/2
print(zz)

dict={'bid':bids_series,'zz':zz}
df1=pd.DataFrame(dict)
print(df1)

#import csv
#with open('dst.csv','wb'）as dstfile:
#    writer=csv.DictWriter(dstfile,fieldnames=header)
#    writer.writeheader()
#    writer.writerows(df1)
#dstfile.close()
df1.to_csv('dst.csv',index=False)

#dff1=df1.groupby('bid').sum('zz')
#print(dff1)


#concat([df1,df2],axis=1)
#pd.merge(df1,df2,on='key')
