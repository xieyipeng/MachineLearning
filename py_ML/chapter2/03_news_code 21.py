# -*- encoding: utf-8 -*-
"""
@File    :   03_news_code 21.py
@Time    :   2020/3/13
@Author  :   xieyipeng
@Review  :   新闻类别识别，朴素贝叶斯
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 
"""
# TODO: code21
from sklearn.datasets import fetch_20newsgroups

# 从互联网下载数据
news = fetch_20newsgroups(subset='all')
print(len(news.data))
print(news.data[0])


# TODO: code22
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    news.data, news.target, test_size=0.25, random_state=33)

# TODO: code23
# 预测
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
vec=CountVectorizer()
X_train=vec.fit_transform(X_train)
X_test=vec.transform(X_test)
mnb=MultinomialNB()
mnb.fit(X_train,y_train)
y_predict=mnb.predict(X_test)

# TODO: code24
# 性能评测
from sklearn.metrics import classification_report
print('精确度(NBC)：',mnb.score(X_test,y_test))
print(classification_report(y_test,y_predict,target_names=news.target_names))

