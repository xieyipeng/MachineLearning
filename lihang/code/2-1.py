"""
正实例点：(3,3),(4,3)
负实例点：(1,1)
"""
import numpy as np

x = np.array([[3, 3], [4, 3], [1, 1]])
y = [1, 1, -1]
eta = 1
w = [0, 0]
b = 0


def is_correct(_x, _y, _w, _b):
    """
    判断是否有分类错误的点
    :param _x: 点集坐标
    :param _y: 真实分类结果
    :param _w: 权重
    :param _b: 偏置
    :return: 是否有错，错误的点的坐标，错误点的序号
    """
    flag = -1
    _wrong = False
    a = 0
    for _i in range(0, len(_y)):
        if _y[_i] * (np.dot(_x[_i], _w) + _b) <= 0:
            flag = _i
            _wrong = True
            a = _i
            break
    return _wrong, x[flag], a + 1


def update(_w, _b, _point, _yi):
    """
    更新参数
    :param _w: 待更新权重
    :param _b: 待更新偏置
    :param _point: 分类错误点坐标
    :param _yi: 分类错误点的真实对应结果
    :return: 更新后的w，b
    """
    _w = _w + eta * _yi * _point
    _b = _b + eta * _yi
    return _w, _b


if __name__ == '__main__':
    while True:
        wrong, point, i = is_correct(x, y, w, b)
        if not wrong:
            print('over')
            break
        print('find the ', i, 'point error: ', point)
        w, b = update(w, b, point, y[(i - 1)])
        print('update w, b: ', w, b)
    print('result: ', w, b)
