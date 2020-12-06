import numpy as np
import operator
import matplotlib.pyplot as plt


def create_data_set():
    """
    test
    :return: gp, lab
    """
    gp = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    lab = ['A', 'A', 'B', 'B']
    return gp, lab


# test
#
# group, labels = create_data_set()
# print(group)
# print(labels)


def classify0(inX, dataSet, labels, k):
    """
    分类算法0
    :param inX: 用于分类的输入向量
    :param dataSet: 训练样本集
    :param labels: 标签向量
    :param k: 最近邻数目
    :return: kNN算法结果
    """
    # 计算距离
    data_set_size = dataSet.shape[0]  # 数据集大小：n
    diffMat = np.tile(inX, (data_set_size, 1)) - dataSet  # 输入inX在各个属性上与各个训练样本距离：n*d （d为属性个数）
    sqDiffMat = diffMat ** 2  # 各元素平方 (n*d)
    sqDistances = sqDiffMat.sum(axis=1)  # 按列求和 (1*n)
    distances = sqDistances ** 0.5  # 开方 (1*n)
    sortedDistIndices = distances.argsort()  # 排序，返回下标 (1*n)

    # 统计距离最小的k个点的标记
    classCount = {}  # 字典类型
    for i in range(k):
        voteI_label = labels[sortedDistIndices[i]]
        classCount[voteI_label] = classCount.get(voteI_label, 0) + 1
    sortedClass_count = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClass_count[0][0]  # 返回最优结果


# predict
#
# group, labels = create_data_set()
# print('predict: ', classify0([0, 0], group, labels, 3))


def file2matrix(filename, dim):
    """
    将文件转换成矩阵
    :param filename: 文件名
    :param dim: 维数，即：属性数
    :return: 样本矩阵，样本标签
    """
    fr = open(filename)  # 打开文件
    arrayOLines = fr.readlines()  # 每次读一行
    numberOfLines = len(arrayOLines)  # 总行数，即：样本数
    returnMat = np.zeros((numberOfLines, dim))  # 构造属性矩阵
    classLabelVector = []  # 构造标签矩阵
    # 填充数据
    index = 0
    for line in arrayOLines:
        # 删除头尾的空格或者换行符
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# test
#
# datingDataMat, datingLabels = file2matrix('./dataset/datingTestSet2.txt', 3)
# print(datingDataMat)
# print(datingLabels[0:20])

def m_draw():
    # 画图
    datingDataMat, datingLabels = file2matrix('./dataset/datingTestSet2.txt', 3)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
    # ax.scatter(datingDataMat[:,1], datingDataMat[:, 2])

    ax2 = fig.add_subplot(122)
    ax2.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
    # ax.scatter(datingDataMat[:,1], datingDataMat[:, 2])
    plt.show()


# test
#
# m_draw()

def auto_norm(dataSet):
    """
    归一化特征值
    :param dataSet:
    :return:
    """
    minValues = dataSet.min(0)
    maxValues = dataSet.max(0)
    ranges = maxValues - minValues
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minValues, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minValues


def test1():
    datingDataMat, datingLabels = file2matrix('./dataset/datingTestSet2.txt', 3)
    normMat, ranges, minVals = auto_norm(datingDataMat)
    print(normMat)
    print(ranges)
    print(minVals)


# test
#
# test1()


def dating_class_test():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('./dataset/datingTestSet2.txt', 3)
    normMat, ranges, minValues = auto_norm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        if classifierResult != datingLabels[i]:
            print("the classifier come back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
            errorCount += 1.0
    print("error rate: %f" % (errorCount / float(numTestVecs)))


# test
#
# dating_class_test()


def classify_person():
    res_list = ['not at all', 'small doses', 'lage doses']
    tats = float(input("percentage time to play games?"))
    miles = float(input("frequent flier miles earned per year?"))
    icecream = float(input("icecream liters?"))
    datingDataMat, datingLabels = file2matrix('./dataset/datingTestSet2.txt', 3)
    normMat, ranges, minValues = auto_norm(datingDataMat)
    inArr = np.array([miles, tats, icecream])
    classifierResult = classify0((inArr - minValues) / ranges, normMat, datingLabels, 3)
    print("you will probably like this person: ", res_list[classifierResult - 1])


# test
#
# classify_person()
