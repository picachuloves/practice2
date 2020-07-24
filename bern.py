import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn.preprocessing import scale
from sklearn import preprocessing
from sklearn import tree
from sklearn.preprocessing import QuantileTransformer


# ##BOOKS
# names = ['rate', 'text']
# df = pd.read_csv('rusBooks.csv', sep=';', names=names)
#
# data_pos = df[df.rate == 'POS']['text']
# data_neg = df[df.rate == 'NEG']['text']


# #MARKET
# df = pd.read_csv('market.csv', keep_default_na=False)
# data_pos = df[df.rating > 3][['comment', 'negative_comment', 'positive_comment']]
# data_neg = df[df.rating <= 3][['comment', 'negative_comment', 'positive_comment']]
#
# data_pos['com'] = data_pos.iloc[:, 0:3].apply(lambda x: ' '.join(x), axis=1)
# data_pos = data_pos['com']
#
# data_neg['com'] = data_neg.iloc[:, 0:3].apply(lambda x: ' '.join(x), axis=1)
# data_neg = data_neg['com']

###

# ###
# #TOXIC
#
# df = pd.read_csv("toxic.csv")
# data_neg = df[df.toxic == 1]['comment']
# data_pos = df[df.toxic == 0]['comment']

###
#NEWS

df = pd.read_json('news.json')
data_pos = df[df.sentiment == 'positive']['text']
data_neg = df[df.sentiment == 'negative']['text']

size_pos = data_pos.shape[0]
size_neg = data_neg.shape[0]
labels = [1] * size_pos + [0] * size_neg
data = data_pos.append(data_neg, ignore_index=True)
data_size = size_neg + size_pos

data_pos = data_pos.head(data_size)
data_neg = data_neg.head(data_size)
data = data_pos.append(data_neg, ignore_index=True)

def preprocess_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub('[^a-zA-Zа-яА-Я]+', ' ', text)
    return text


datap = [preprocess_text(t) for t in data]

###
countVectorizer = CountVectorizer()
vectors = countVectorizer.fit_transform(datap)

###
#COUNT
x_train, x_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.3, random_state=1)

print('Bern')

BernNB = BernoulliNB()
BernNB.fit(x_train, y_train)
y_expect = y_test
y_predict = BernNB.predict(x_test)

print(cross_val_score(BernNB, x_train, y_train, cv=5))
print(accuracy_score(y_expect, y_predict))

print(recall_score(y_expect, y_predict))

print(precision_score(y_expect, y_predict))

print(f1_score(y_expect, y_predict))


# print("scale")
#
# scaler = StandardScaler(with_mean=False)
# x_train_array = x_train.toarray()
# print(x_train_array.shape)
# print("---")
#
# print(x_train_array.mean(axis=0))
# print(x_train_array.std(axis=0))
# scaled_x_train_array = scaler.fit_transform(x_train_array)
# print(scaled_x_train_array.mean(axis=0))
# print(scaled_x_train_array.std(axis=0))
# x_test_array = x_test.toarray()
# scaled_x_test_array = scale(x_test_array)

###
#
# print('Bern Scaled')
#
# BernNBScaled = BernoulliNB()
# BernNBScaled.fit(scaled_x_train_array, y_train)
# y_predict = BernNBScaled.predict(scaled_x_test_array)
#
# print(cross_val_score(BernNBScaled, scaled_x_train_array, y_train, cv=5))
# print(accuracy_score(y_expect, y_predict))
#
# print(recall_score(y_expect, y_predict))
#
# print(precision_score(y_expect, y_predict))
#
# print(f1_score(y_expect, y_predict))
#
# ###
#
# print('Bern Quantile')
#
# qt = QuantileTransformer(n_quantiles=10, random_state=0)
# qt.fit_transform(X)
#
# BernNBScaled = BernoulliNB()
# BernNBScaled.fit(scaled_x_train_array, y_train)
# y_predict = BernNBScaled.predict(scaled_x_test_array)
#
# print(cross_val_score(BernNBScaled, scaled_x_train_array, y_train, cv=5))
# print(accuracy_score(y_expect, y_predict))
#
# print(recall_score(y_expect, y_predict))
#
# print(precision_score(y_expect, y_predict))
#
# print(f1_score(y_expect, y_predict))

