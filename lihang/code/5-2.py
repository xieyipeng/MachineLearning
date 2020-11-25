import numpy as np
from math import log2
import

D = np.array([
    ['青年', '否', '否', '一般', '否'],
    ['青年', '否', '否', '好', '否'],
    ['青年', '是', '否', '好', '是'],
    ['青年', '是', '是', '一般', '是'],
    ['青年', '否', '否', '一般', '否'],
    ['中年', '否', '否', '一般', '否'],
    ['中年', '否', '否', '好', '否'],
    ['中年', '是', '是', '好', '是'],
    ['中年', '否', '是', '非常好', '是'],
    ['中年', '否', '是', '非常好', '是'],
    ['老年', '否', '是', '非常好', '是'],
    ['老年', '否', '是', '好', '是'],
    ['老年', '是', '否', '好', '是'],
    ['老年', '是', '否', '非常好', '是'],
    ['老年', '否', '否', '一般', '否']
], dtype=np.str)

MAX = 10

t_k_ = np.array([3, 2, 2, 3, 2])  # 各个域的大小
t_k = np.array([
    ['青年', '中年', '老年'],
    ['是', '否'],
    ['是', '否'],
    ['一般', '好', '非常好'],
    ['是', '否']
])

c_k = np.zeros((len(t_k_), MAX), dtype=np.int)

D_ = len(D)  # 样本数
t_ = len(D[0]) - 1  # 特征数

HDA = np.zeros(t_)


def get_HDA():
    """
    计算经验条件熵
    :return: None
    """
    global HDA, D, t_, t_k, t_k_, c_k, D_
    for ti in range(t_):  # 待计算特征A下标
        res = 0
        for yi in range(t_k_[t_]):  # 对于每一个分类
            temp = 0
            for ki in range(t_k_[ti]):  # 对于该特征的每一个取值
                for di in range(D_):  # 对于每一个数据集
                    if D[di][ti] == t_k[ti][ki] and D[di][t_] == t_k[t_][yi]:
                        temp += 1
                res += (c_k[ti][ki] / D_) * (t(temp, c_k[ti][ki]))
                temp = 0
        HDA[ti] = res


def get_C_k():
    """
    统计各个特征的各种取值的个数
    :return: None
    """
    global D_, D, t_, c_k, t_k_, t_k
    for ti in range(t_ + 1):  # 待计算特征A下标
        for ki in range(t_k_[ti]):  # 对于该特征的每一个取值
            for di in D:
                if di[ti] == t_k[ti][ki]:
                    c_k[ti][ki] += 1


def get_HD(ti, _d_):
    """
    计算经验熵
    :param ti: 最终分类所在下标
    :param _d_: 分母，即数据集大小
    :return: 计算结果，即数据集D的经验熵
    """
    global D, D_, t_k_, t_, c_k
    res = 0
    for i in range(t_k_[ti]):
        res += t(c_k[ti][i], _d_)
    return res


def t(_ck, _n):
    """
    计算log
    :param _ck: 出现频次
    :param _n: 总数
    :return: 计算结果
    """
    if _ck == 0:
        return 0
    temp = _ck / _n
    return -temp * log2(temp)


def main():
    """
    主函数
    :return: None
    """
    global t_, D_
    get_C_k()
    HD = get_HD(t_, D_)
    get_HDA()
    res = 0
    j = 0
    for i in range(t_):
        if HD - HDA[i] >= res:
            res = HD - HDA[i]
            j = i
    print('选择特征为A', j + 1, '信息增益为', res)


if __name__ == '__main__':
    main()
