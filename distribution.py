import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats



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

###
vectorizer = CountVectorizer()
vectors1 = vectorizer.fit_transform(datap)
vectors2 = vectorizer.fit_transform(data)
