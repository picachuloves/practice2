import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

#BOOKS
# names = ['rate', 'text']
# df = pd.read_csv('rusBooks.csv', sep=';', names=names)
#
# data_pos = df[df.rate == 'POS']['text']
# data_neg = df[df.rate == 'NEG']['text']

###
#MARKET

# df = pd.read_csv('market.csv', keep_default_na=False)
# data_pos = df[df.rating > 3][['comment', 'negative_comment', 'positive_comment']]
# data_neg = df[df.rating <= 3][['comment', 'negative_comment', 'positive_comment']]
#
# data_pos['com'] = data_pos.iloc[:, 0:3].apply(lambda x: ' '.join(x), axis=1)
# data_pos = data_pos['com']
#
# data_neg['com'] = data_neg.iloc[:, 0:3].apply(lambda x: ' '.join(x), axis=1)
# data_neg = data_neg['com']

# #NEWS

# df = pd.read_json('news.json')
# data_pos = df[df.sentiment == 'positive']['text']
# data_neg = df[df.sentiment == 'negative']['text']

###

###
#TOXIC

df = pd.read_csv("toxic.csv")
data_neg = df[df.toxic == 1]['comment']
data_pos = df[df.toxic == 0]['comment']

###

size_pos = data_pos.shape[0]
size_neg = data_neg.shape[0]
labels = [1] * size_pos + [0] * size_neg

data_size = size_neg + size_pos
data_pos = data_pos.head(data_size)
data_neg = data_neg.head(data_size)
data = data_pos.append(data_neg, ignore_index=True)

def preprocess_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub('[^a-zA-Zа-яА-Я]+', ' ', text)
    return text


datap = [preprocess_text(t) for t in data]

vectorizer = CountVectorizer()
vectors1 = vectorizer.fit_transform(datap)
vectors2 = vectorizer.fit_transform(data)

print("---")
print(vectors1.shape)
print(vectors2.shape)

x_train, x_test, y_train, y_test = train_test_split(vectors1, labels, test_size=0.3, random_state=1)
y_expect = y_test

###
# scaler = StandardScaler(with_mean=False)
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)

###

print('KNN')
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train, y_train)
svm_predict = clf.predict(x_test)

print(cross_val_score(clf, x_train, y_train, cv=5))
print(accuracy_score(y_expect, svm_predict))
print(recall_score(y_expect, svm_predict))
print(precision_score(y_expect, svm_predict))
print(f1_score(y_expect, svm_predict))