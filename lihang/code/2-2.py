"""
正实例点：(3,3),(4,3)
负实例点：(1,1)
"""
import numpy as np

x = np.array([[3, 3], [4, 3], [1, 1]])
n = len(x)
y = [1, 1, -1]
eta = 1
w = [0, 0]
b = 0
alpha = np.zeros(n, dtype=np.int)


def is_correct(_point, _label, _row, _g):
    """
    判断是否分类正确
    :param _point: 判断点
    :param _label: 该点真实标签
    :param _row: 该点序号
    :param _g: Gram矩阵
    :return: 对偶形式计算结果
    """
    global b
    _wrong = False
    temp = 0
    for _j in range(n):
        temp += eta * alpha[_j] * _label[_j] * _g[_j][_row]
    temp += b
    temp *= _label[_row]
    return temp


def update(_i, _y):
    """
    更新参数
    :param _i: 序号
    :param _y: 真实标签
    :return: None
    """
    global b, alpha
    alpha[_i] += eta
    b += eta * _y[_i]


def main():
    ok = False
    G = np.zeros((n, n), dtype=np.int)  # 对称阵
    for i in range(0, 3):
        for j in range(0, 3):
            G[i][j] = x[i][0] * x[j][0] + x[i][1] * x[j][1]
    while not ok:
        for i in range(n):
            if is_correct(x[i], y, i, G) <= 0:
                update(i, y)
                print(alpha, b)
                break
            elif i == n - 1:
                ok = True
                print(alpha, b)


if __name__ == '__main__':
    main()
