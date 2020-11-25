# -*- encoding: utf-8 -*-
"""
@File    :   01_tumor_drawing.py
@Time    :   2020/3/12
@Author  :   xieyipeng
@Review  :   肿瘤良/恶性展示图
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 
"""
import pandas as pd
import matplotlib.pyplot as plt

df_test = pd.read_csv(r'../src/p38/graph.csv')

# 选取‘Clump Thickness’和‘Cell Size’作为特征，构建测试集中的正负分类样本
# TODO:选取所有的type == 0 的良性肿瘤数据
df_test_negative = df_test.loc[df_test['class'] == 2][['clump thickness', 'uniformity of cell size']]
# TODO:选取所有的type == 1 的恶性肿瘤数据
df_test_positive = df_test.loc[df_test['class'] == 4][['clump thickness', 'uniformity of cell size']]

# 良性标记为红色
plt.scatter(df_test_negative['clump thickness'], df_test_negative['uniformity of cell size'], marker='o', s=200, c='red')
# 恶性标记为黑色
plt.scatter(df_test_positive['clump thickness'], df_test_positive['uniformity of cell size'], marker='x', s=150, c='black')

# TODO:绘制x，y轴的说明
plt.xlabel('clump thickness')
plt.ylabel('uniformity of cell size')
# 显示图
plt.show()
