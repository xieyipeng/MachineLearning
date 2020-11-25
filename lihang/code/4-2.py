import numpy as np

"""
已知条件：
"""
T = np.array([[1, 'S', -1], [1, 'M', -1], [1, 'M', 1], [1, 'S', 1], [1, 'S', -1],
              [2, 'S', -1], [2, 'M', -1], [2, 'M', 1], [2, 'L', 1], [2, 'L', 1],
              [3, 'L', 1], [3, 'M', 1], [3, 'M', 1], [3, 'L', 1], [3, 'L', -1]], dtype=np.str)

Max = 80

Y = np.array([-1, 1], dtype=np.str)
A1 = np.array([1, 2, 3], dtype=np.str)
A2 = np.array(['S', 'M', 'L'], dtype=np.str)

"""
演算：
"""

A = np.array([len(A1), len(A2)])
A_fill = np.array([[1, 2, 3], ['S', 'M', 'L']], dtype=np.str)

N = len(T)
T_j = len(T[0]) - 1  # 特征数

_lambda = 1

p_ij = np.zeros(shape=(len(Y), T_j, Max))

global y

x = np.array([2, 'S'])  # 输入


def P_Y_ck():
    """
    对数据集来说，不同分类的概率
    :return: 不同分类的概率
    """
    global T, N, _lambda, T_j
    _y_1 = 0
    _y_2 = 0
    for i in range(N):
        if T[i][2] == '-1':
            _y_1 += 1
        else:
            _y_2 += 1
    return _y_1, _y_2


def filling_P():
    """
    填充概率矩阵
    :return: None
    """
    global T, N, Y, A_fill, p_ij, y, _lambda, T_j, A
    for yi in range(len(Y)):
        for xi in range(T_j):
            for _l in range(A[xi]):
                temp = 0
                for t in range(N):
                    if T[t][2] == Y[yi] and T[t][xi] == A_fill[xi][_l]:
                        temp += 1
                temp += _lambda
                p_ij[yi][xi][_l] = temp / (y[yi] + A[xi])


def out():
    """
    输出概率矩阵
    :return: None
    """
    print('计算结果:')
    for yi in range(len(Y)):
        for xi in range(T_j):
            for _l in range(A[xi]):
                print(p_ij[yi][xi][_l], end=' ')
            print()


def prediction():
    """
    进行预测
    :return: 最可能结果及其概率
    """
    global x, _lambda, T_j
    m = np.zeros(T_j, dtype=np.int)
    for i in range(T_j):
        for j in range(A[i]):
            if A_fill[i][j] == x[i]:
                m[i] = j
    _res = 0
    _res_y = 0
    for yi in range(len(Y)):
        p = 1
        for xi in range(T_j):
            p *= p_ij[yi][xi][m[xi]]
        p *= (y[yi] + _lambda) / (N + T_j * _lambda)
        if p >= _res:
            _res = p
            _res_y = Y[yi]
    return _res, _res_y


def main():
    global y
    y_1, y_2 = P_Y_ck()
    y = [y_1, y_2]
    filling_P()
    # out()
    res, res_y = prediction()
    print('预测结果: y = ', res_y, '概率: p = ', res)


if __name__ == '__main__':
    main()
