# 良/恶性乳腺癌肿瘤预测
import pandas as pd
# 导入matplotlib工具包的pyplot并化简为plt
import matplotlib.pyplot as plt
# 导入numpy工具包，命名为np`a
import numpy as np
# 导入sklearn中的逻辑斯蒂回归分类器
from sklearn.linear_model import LogisticRegression

# 调用pandas工具包的read_csv函数/模块，传入训练文件地址参数，获得返回的数据并存放至de_train
df_train = pd.read_csv('/home/xieyipeng/Documents/MachineLearning/py_ML/src/p30/breast-cancer-train.csv')
df_test = pd.read_csv('/home/xieyipeng/Documents/MachineLearning/py_ML/src/p30/breast-cancer-test.csv')

# 选取‘Clump Thickness’和‘Cell Size’作为特征，构建测试集中的正负分类样本
# TODO:选取所有的type == 0 的良性肿瘤数据
df_test_negative = df_test.loc[df_test['Type'] == 0][['Clump Thickness', 'Cell Size']]
# TODO:选取所有的type == 1 的恶性肿瘤数据
df_test_positive = df_test.loc[df_test['Type'] == 1][['Clump Thickness', 'Cell Size']]

# 良性标记为红色
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
# 恶性标记为黑色
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')

# TODO:绘制x，y轴的说明
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
# 显示图
plt.show()
# TODO: 随机绘制的二类分类器
# 利用numpy中的random函数，随机采样直线的截距和系数
intercept = np.random.random([1])
coef = np.random.random([2])
lx = np.arange(0, 12)  # lx [ 0  1  2  3  4  5  6  7  8  9 10 11]
ly = (-intercept - lx * coef[0]) / coef[1]
# 绘制一条随机直线
plt.plot(lx, ly, c='yellow')
# 绘制图
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
# 显示图
plt.show()
# TODO: 绘制10个训练集获得的二类分类器
lr = LogisticRegression()
# 使用前十条训练样本学习直线的系数和截距
lr.fit(df_train[['Clump Thickness', 'Cell Size']][:10], df_train['Type'][:10])
print('Testing accucracy(10 training samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type']))
intercept = lr.intercept_
coef = lr.coef_[0, :]
ly = (-intercept - lx * coef[0]) / coef[1]
# 绘制图1-4
plt.plot(lx, ly, c='green')
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()
# TODO: 绘制所有训练集获得的二类分类器
lr = LogisticRegression()
# 使用所有的训练样本学习直线的系数和截距
lr.fit(df_train[['Clump Thickness', 'Cell Size']], df_train['Type'])
print('Testing accucracy(all training samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type']))
intercept = lr.intercept_
coef = lr.coef_[0, :]
ly = (-intercept - lx * coef[0]) / coef[1]
# 绘制图1-5
plt.plot(lx, ly, c='blue')
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()
