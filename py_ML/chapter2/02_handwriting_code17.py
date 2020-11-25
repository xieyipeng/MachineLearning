# -*- encoding: utf-8 -*-
"""
@File    :   02_handwriting_code17.py
@Time    :   2020/3/12
@Author  :   xieyipeng
@Review  :   手写体识别
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 
"""

# TODO： code17
# 导入手写体数字加载器
from sklearn.datasets import load_digits

digits = load_digits()
print(digits.data.shape)

# TODO: code18
# 数据集分割
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.25, random_state=33)
# print(y_train.shape)
# print(y_test.shape)

# TODO: code19
# 手写体识别
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
lsvc = LinearSVC(max_iter=10000)
# 模型训练
lsvc.fit(X_train, y_train)
y_predict = lsvc.predict(X_test)
# print(y_predict.shape)

# TODO: code20
# 性能评估
from sklearn.metrics import classification_report

print('精确度(SVC): ', lsvc.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names=digits.target_names.astype(str)))
