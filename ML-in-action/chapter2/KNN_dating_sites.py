#!/usr/bin/env python
# coding: utf-8

# In[16]:


# KNN.py
from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# In[17]:


group, labels = createDataSet()
group, labels


# In[54]:


def classify0(inX, dataSet, labels, k):
    
    # 计算距离
    data_set_size = dataSet.shape[0]
    diffMat = tile(inX, (data_set_size, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    
    # 选择距离最小的k个点
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClasscount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClasscount[0][0]


# In[39]:


# test:
print('group.shape:', group.shape)

data_set_size = group.shape[0]
data_set_size


diffMat = tile(array([5,6]), (data_set_size, 1)) - group
print('diffMat:\n', diffMat)
sqdiffMat = diffMat**2
sqDistances = sqdiffMat.sum(axis=1)

print('sqDiffMat:\n', sqDistances)

sortedDistIndices = sqDistances.argsort()

print('sortedDiffMat:\n', sortedDistIndices)

classCount={}

voteIlabel = labels[sortedDistIndices[0]]
classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

print('class_count:\n', classCount)


# In[55]:


# predict
classify0([0, 0], group, labels, 3)


# ### K-近邻算法优化约会网站配对效果

# In[56]:


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        # 删除头尾的空格或者换行符
        line = line.strip()
        
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# In[57]:


datingDataMat, datingLabels = file2matrix('./dataset/datingTestSet2.txt')
datingDataMat, datingLabels[0:20]


# In[58]:


import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.scatter(datingDataMat[:,1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
# ax.scatter(datingDataMat[:,1], datingDataMat[:, 2])

ax2 = fig.add_subplot(122)
ax2.scatter(datingDataMat[:,0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
# ax.scatter(datingDataMat[:,1], datingDataMat[:, 2])

plt.show()


# In[59]:


def autoNorm(dataSet):
    minValues = dataSet.min(0)
    maxValues = dataSet.max(0)
    ranges = maxValues - minValues
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minValues, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minValues


# In[60]:


normMat, ranges, minVals = autoNorm(datingDataMat)
normMat, ranges, minVals


# In[71]:


def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('./dataset/datingTestSet2.txt')
    normMat, ranges, minValues = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
        if (classifierResult != datingLabels[i]):
            print("the classifier come back with: %d, the real answer is: %d"%(classifierResult, datingLabels[i]))
            errorCount += 1.0
    print("error rate: %f"%(errorCount / float(numTestVecs)))

datingClassTest()


# In[76]:


def classifyPerson():
    res_list = ['not at all', 'small doses', 'lage doses']
    tats = float(input("percentage time to play games?"))
    miles = float(input("frequent flier miles earned per year?"))
    icecream = float(input("icecream liters?"))
    datingDataMat, datingLabels = file2matrix('./dataset/datingTestSet2.txt')
    normMat, ranges, minValues = autoNorm(datingDataMat)
    inArr = array([miles, tats, icecream])
    classifierResult = classify0((inArr - minValues) / ranges, normMat, datingLabels, 3)
    print("you will probably like this person: ", res_list[classifierResult -1])

classifyPerson()


# In[ ]:




