#-*- coding:utf-8 -*-
import json
with open("./review2.json") as data_file:    
    data = json.load(data_file)

#print(data[1]['stars'])
#print(data[1]['user_id'])
#print(data[1]['text'])
print(len(data));

import pandas as pd



raw_docs = []
urls=[]
labels = []
bids= []
rids= []
#sen_dic={"positive":1,"negative":-1,"neutral":0}
for i in range(len(data)):
    raw_docs.append(data[i]['text'])
    urls.append(data[i]['user_id'])
    labels.append(data[i]['stars'])
    bids.append(data[i]['business_id'])
    rids.append(data[i]['review_id'])
#print (raw_docs[0])


df1=pd.DataFrame(bids,labels) 
print(df1)

import nltk
from nltk.tokenize import word_tokenize
tokenized_docs = [word_tokenize(doc) for doc in raw_docs]

import re
import string
regex = re.compile('[%s]' % re.escape(string.punctuation)) #see documentation here: http://docs.python.org/2/library/string.html

tokenized_docs_no_punctuation = []

for review in tokenized_docs:
    new_review = []
    for token in review:
        new_token = regex.sub(u'', token)
        if not new_token == u'':
            new_review.append(new_token)
    
    tokenized_docs_no_punctuation.append(new_review)
    

#print ((tokenized_docs_no_punctuation[2]))


from nltk.corpus import stopwords

tokenized_docs_no_stopwords = []

for doc in tokenized_docs_no_punctuation:
    new_term_vector = []
    for word in doc:
        if not word in stopwords.words('english'):
            new_term_vector.append(word)
    
    tokenized_docs_no_stopwords.append(new_term_vector)

#print("...")
#repr(tokenized_docs_no_stopwords[2]).decode("UTF-8")


from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

porter = PorterStemmer()
snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()

preprocessed_docs = []

for doc in tokenized_docs_no_stopwords:
    final_doc = []
    for word in doc:
        final_doc.append(porter.stem(word))
        #final_doc.append(snowball.stem(word))
        #final_doc.append(wordnet.lemmatize(word))
    
    preprocessed_docs.append(final_doc)

#print (preprocessed_docs[2])

pre_svm=[]
for i in range(len(preprocessed_docs)):
     string=' '.join(preprocessed_docs[i])
     pre_svm.append(string)

print("svm....")
print (pre_svm[0])
print(len(pre_svm[0]))

star2 = []
for i in range(len(labels)):
    if labels[i] <= 3:
        star2.append(0)
    else:
        star2.append(1)
print (star2)

star3 = []
print (labels)
for i in range(len(labels)):
    star = 1.0/(labels[i]+1)
    star3.append(star)
print (star3)

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import dump_svmlight_file
import numpy as np
import os

#y = sentiments
#y = [0 for x in range(1,101)]
i = 0

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(pre_svm)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)

print("xxxxxx")
print(X[0])
print("yyyyyy")
print(tfidf[0])
#print(len(y))

f = open('./review4.json', 'wb')
dump_svmlight_file(tfidf, star2, f, zero_based=False)
f.close()


f = open('./review3.json', 'wb')
dump_svmlight_file(tfidf, star2, f, zero_based=False)
f.close()

#print X   
#print repr(vectorizer.get_feature_names()).decode("unicode-escape")
#print (tfidf[0])

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)

# Load training data
data = spark.read.format("libsvm") \
    .load("/home/yk/yelp4/review4.json")

# Split the data into train and test
splits = data.randomSplit([0.9, 0.1], 12)
train = splits[0]
test = splits[1]

print("xxxxx")
#train.show(5)
#print(data.count())


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

# Select example rows to display.
result2=predictions2.select("prediction")
result2.show(5)

