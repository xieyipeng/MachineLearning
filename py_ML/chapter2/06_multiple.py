# -*- encoding: utf-8 -*-
"""
@File    :   06_multiple.py
@Time    :   2020/3/14
@Author  :   xieyipeng
@Review  :   集成模型
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 
"""

# TODO: code32
# 集成模型预测
import pandas
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier

titanic = pandas.read_csv('../src/p57/titanic.csv')
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
X['age'].fillna(X['age'].mean(), inplace=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=33)
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
# print(vec.feature_names_)
X_test = vec.transform(X_test.to_dict(orient='record'))

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_y_predict = dtc.predict(X_test)

# 使用随即森林分类器进行预测
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_y_predict = rfc.predict(X_test)

# 使用梯度上升决策树进行集成模型训练
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_y_predict = gbc.predict(X_test)

# TODO: code33
# 集成模型性能预测
from sklearn.metrics import classification_report

print('精确度(decision_tree):', dtc.score(X_test, y_test))
print(classification_report(dtc_y_predict, y_test))

print('精确度(rfc):', rfc.score(X_test, y_test))
print(classification_report(rfc_y_predict, y_test))

print('精确度(gbc):', gbc.score(X_test, y_test))
print(classification_report(gbc_y_predict, y_test))
