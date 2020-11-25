# -*- encoding: utf-8 -*-
"""
@File    :   04_iris_code25.py
@Time    :   2020/3/13
@Author  :   xieyipeng
@Review  :   鸢尾 - k近邻
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 
"""
# TODO: code 25
# 读取iris数据集
from sklearn.datasets import load_iris

iris = load_iris()
# print(iris.data.shape)
# print(iris.DESCR)

# TODO: code26
# 数据集分割
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.25, random_state=33)

# TODO: code27
# 预测
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
y_predict = knc.predict(X_test)

# TODO: code28
# 性能评测
print('精确度(K)：',knc.score(X_test,y_test))
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict,target_names=iris.target_names))