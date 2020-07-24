import pandas as pd
import re
import math

df1 = pd.read_csv("market.csv", sep=',')
print(df1.head(6))
print(df1.columns)
df1_1 = df1[['comment', 'rating', 'negative_comment', 'positive_comment']]
print(df1_1.iloc[478])

neg_coms = df1[math.isnan(df1.negative_comment) != True][['negative_comment']]
pos_coms = df1[['positive_comment']]
coms = df1[['comment']]
rates = df1[['rating']]
print('d')


def preprocess_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    return text
