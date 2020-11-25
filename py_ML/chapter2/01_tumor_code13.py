# -*- encoding: utf-8 -*-
"""
@File    :   01_tumor_code13.py
@Modify Time      @Author
 ---------        -------
 2020/3/11       xieyipeng
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 
"""

# p38
# 线性分类器

import pandas
import numpy

# TODO： code13
# 数据集地址： https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
# 创建特征列表 编号id/肿块厚度/细胞大小均匀性/细胞形状均匀性/边际附着力/单上皮细胞大小/bare nuclei/乏味的染色质/正常的核仁/有丝分裂/性状
column_names = ['id', 'clump thickness', 'uniformity of cell size', 'uniformity of cell shape', 'marginal adhesion',
                'single epithelial cell size', 'bare nuclei', 'bland chromatin', 'normal nucleoli', 'mitoses', 'class']
data = pandas.read_csv('../src/p38/breast-cancer-wisconsin.data', names=column_names)
# 缺失值替换
data = data.replace(to_replace='?', value=numpy.nan)
# 丢弃有缺失值的数据
data = data.dropna(how='any')
print(data.shape)
# print(data)

# TODO: code14
# 25%作为测试集，75%作为训练集


from sklearn.model_selection import train_test_split

# X_train,X_test, y_train, y_test =sklearn.model_selection.train_test_split(train_data,train_target,test_size=0.4,
# random_state=0,stratify=y_train)
X_train, X_test, y_train, y_test = train_test_split(
    data[column_names[1:10]], data[column_names[10]], test_size=0.25, random_state=33)
# 检查各数量和类别分布
# print(y_train.value_counts())
# print(y_test.value_counts())


# TODO： code15
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

# 标准化特征
ss = StandardScaler()
# print(X_train.info())
# print(X_test.info())
# print(type(X_train))
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
# print(X_train.shape)
# print(X_test.shape)
# 初始化逻辑斯蒂函数和梯度下降分类方法
lr = LogisticRegression()
sgdc = SGDClassifier()
# 开始训练
lr.fit(X_train, y_train)
sgdc.fit(X_train, y_train)
# 预测
lr_predict = lr.predict(X_test)
sgdc_predict = sgdc.predict(X_test)

# print(lr_predict.shape)
# print(sgdc_predict.shape)

# TODO: code16
# 性能分析
from sklearn.metrics import classification_report

# 逻辑斯蒂回归模型
print('准确率(LR): ', lr.score(X_test, y_test))
# print(classification_report(y_test, lr_predict, target_names=['Benign', 'Malignant']))
print(classification_report(y_test, lr_predict, target_names=['良性', '恶性']))
# SGDC
print('准确率(SGDC)：',sgdc.score(X_test,y_test))
print(classification_report(y_test,sgdc_predict,target_names=['良性', '恶性']))