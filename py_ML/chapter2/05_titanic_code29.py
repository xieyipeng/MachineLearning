# -*- encoding: utf-8 -*-
"""
@File    :   05_titanic_code29.py
@Time    :   2020/3/13
@Author  :   xieyipeng
@Review  :   泰坦尼克号
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 
"""

# TODO: code29
# 导入数据
import pandas

titanic = pandas.read_csv('../src/p57/titanic.csv')
# print(titanic.head())
# print(titanic.info())

# TODO： code30
# 数据预处理
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
# print(X.info())

# 补充age缺失
X['age'].fillna(X['age'].mean(), inplace=True)
print(X.info())

# 数据分割
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=33)
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
print(vec.feature_names_)
X_test = vec.transform(X_test.to_dict(orient='record'))
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_predict = dtc.predict(X_test)

# TODO: ode31
# 性能评测
from sklearn.metrics import classification_report

print(dtc.score(X_test, y_test))
print(classification_report(y_predict, y_test, target_names=['died', 'survived']))
