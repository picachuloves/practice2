import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

###
# #BOOKS
# names = ['rate', 'text']
# df = pd.read_csv('rusBooks.csv', sep=';', names=names)
#
# data_pos = df[df.rate == 'POS']['text']
# data_neg = df[df.rate == 'NEG']['text']

###
# #MARKET
#
# df = pd.read_csv('market.csv', keep_default_na=False)
# data_pos = df[df.rating > 3][['comment', 'negative_comment', 'positive_comment']]
# data_neg = df[df.rating <= 3][['comment', 'negative_comment', 'positive_comment']]
#
# data_pos['com'] = data_pos.iloc[:, 0:3].apply(lambda x: ' '.join(x), axis=1)
# data_pos = data_pos['com']
#
# data_neg['com'] = data_neg.iloc[:, 0:3].apply(lambda x: ' '.join(x), axis=1)
# data_neg = data_neg['com']

# ###
#TOXIC

df = pd.read_csv("toxic.csv")
data_neg = df[df.toxic == 1]['comment']
data_pos = df[df.toxic == 0]['comment']

###
#NEWS
#
# df = pd.read_json('news.json')
# data_pos = df[df.sentiment == 'positive']['text']
# data_neg = df[df.sentiment == 'negative']['text']

###
pos_num = data_pos.shape[0]
neg_num = data_neg.shape[0]
data_num = neg_num + pos_num
data_size = min(pos_num, neg_num)

data_pos = data_pos.head(data_size)
data_neg = data_neg.head(data_size)
data = data_pos.append(data_neg, ignore_index=True)

def preprocess_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub('[^a-zA-Zа-яА-Я]+', ' ', text)
    return text

data_prep = [preprocess_text(t) for t in data]

countVectorizer = CountVectorizer()
tfidfVectorizer = TfidfVectorizer()

c_vectors = countVectorizer.fit_transform(data)
c_vectors_p = countVectorizer.fit_transform(data_prep)
t_vectors = tfidfVectorizer.fit_transform(data)
t_vectors_p = tfidfVectorizer.fit_transform(data_prep)




print('end')

labels = [1] * data_size + [0] * data_size