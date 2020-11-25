import numpy as np

T = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
n = T.shape[0]
k = T.shape[1]


class Node(object):
    """
    节点类，用于创建二叉树
    """
    def __init__(self, item=None):
        self.elem = item
        self.l_child = None
        self.r_child = None


def traversal(node, mes):
    """
    二叉树遍历
    :param node: 根节点
    :param mes: 节点信息
    :return: None
    """
    if node is None:
        print('none')
        return
    print(node.elem, mes)
    traversal(node.l_child, 'l')
    traversal(node.r_child, 'r')


def build_kd_tree(_node, _depth, m_set):
    """
    创建二叉树
    :param _node: 待插入node节点
    :param _depth: 该node深度
    :param m_set: 待选定数据坐标集合
    :return: None
    """
    global k
    if len(m_set) == 0:
        return
    cut_dim = _depth % k
    len_set = len(m_set)
    mid = len_set // 2
    m_sort_set = sorted(m_set, key=lambda x: x[cut_dim])
    # print(m_sort_set)
    point = m_sort_set[mid]
    # print('choose: ', point)
    _node.elem = point
    # print(_node.elem)
    l_set = []
    r_set = []
    for _i in range(0, mid):
        l_set.append(m_sort_set[_i])
    for _i in range(mid + 1, len(m_set)):
        r_set.append(m_sort_set[_i])

    if len(l_set) is not 0:
        _node.l_child = Node()
        build_kd_tree(_node.l_child, _depth + 1, l_set)
    if len(r_set) is not 0:
        _node.r_child = Node()
        build_kd_tree(_node.r_child, _depth + 1, r_set)


def main():
    """
    主函数
    :return: None
    """
    kd_node = Node()
    build_kd_tree(kd_node, 0, T)
    traversal(kd_node, 'root')


if __name__ == '__main__':
    main()


