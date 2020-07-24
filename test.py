import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
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

names = ['rate', 'text']
df = pd.read_csv('rusBooks.csv', sep=';', names=names)

data_pos = df[df.rate == 'POS']['text']
data_neg = df[df.rate == 'NEG']['text']

size_pos = data_pos.shape[0]
size_neg = data_neg.shape[0]
labels = [1] * size_pos + [0] * size_neg
data = data_pos.append(data_neg, ignore_index=True)
data_size = size_neg + size_pos


def preprocess_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub('[^a-zA-Zа-яА-Я]+', ' ', text)
    return text


datap = [preprocess_text(t) for t in data]

vectorizer = CountVectorizer()
vectors1 = vectorizer.fit_transform(datap)
vectors2 = vectorizer.fit_transform(data)


print('Bern')

x_train, x_test, y_train, y_test = train_test_split(vectors1, labels, test_size=0.3, random_state=1)

BernNB = BernoulliNB()
BernNB.fit(x_train, y_train)
y_expect = y_test
y_predict = BernNB.predict(x_test)

print(cross_val_score(BernNB, x_train, y_train, cv=5))
print(accuracy_score(y_expect, y_predict))

print(recall_score(y_expect, y_predict))

print(precision_score(y_expect, y_predict))

print(f1_score(y_expect, y_predict))


print('Bern2')

x_train, x_test, y_train, y_test = train_test_split(vectors2, labels, test_size=0.3, random_state=1)

BernNB = BernoulliNB()
BernNB.fit(x_train, y_train)
y_expect = y_test
y_predict = BernNB.predict(x_test)

print(cross_val_score(BernNB, x_train, y_train, cv=5))
print(accuracy_score(y_expect, y_predict))

print(recall_score(y_expect, y_predict))

print(precision_score(y_expect, y_predict))

print(f1_score(y_expect, y_predict))

print("---")
print(vectors1.shape)
print(vectors2.shape)

print('Bern Scaled')

print("scale")

scaler = StandardScaler(with_mean=False)
x_train_array = x_train.toarray()
x_test_array = x_test.toarray()
print("---")

print(x_train_array.mean(axis=0))
print(x_train_array.std(axis=0))
scaled_x_train_array = scaler.fit_transform(x_train_array)
print(scaled_x_train_array.mean(axis=0))
print(scaled_x_train_array.std(axis=0))

scaled_x_test_array = scaler.fit_transform(x_test_array)

BernNBScaled = BernoulliNB()
BernNBScaled.fit(scaled_x_train_array, y_train)
y_predict = BernNBScaled.predict(scaled_x_test_array)

print(cross_val_score(BernNBScaled, scaled_x_train_array, y_train, cv=5))
print(accuracy_score(y_expect, y_predict))

print(recall_score(y_expect, y_predict))

print(precision_score(y_expect, y_predict))

print(f1_score(y_expect, y_predict))

print('Bern Quantile')

qt = QuantileTransformer(n_quantiles=10, random_state=0)

q_x_train_array = qt.fit_transform(x_train_array)
q_x_test_array = qt.fit_transform(x_test_array)

BernNBScaled = BernoulliNB()
BernNBScaled.fit(q_x_train_array, y_train)
y_predict = BernNBScaled.predict(q_x_test_array)

print(cross_val_score(BernNBScaled, q_x_train_array, y_train, cv=5))
print(accuracy_score(y_expect, y_predict))

print(recall_score(y_expect, y_predict))

print(precision_score(y_expect, y_predict))

print(f1_score(y_expect, y_predict))
