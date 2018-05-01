import numpy as np
from math import log
import operator


# 计算给定数据集的香农熵
# 划分数据集的大原则是：将无序的数据变得更加有序。我们可以使用多种方法划分数据集，但是每种方法都有各自的优缺点。
# 组织杂乱无章数据的一种方法就是使用信息论度量信息，信息论是量化处理信息的分支科学。
# 我们可以在划分数据之前或之后使用信息论量化度量信息的内容。
def calcshannonent(dataset):
    numentries = len(dataset)  # 获取数据行数
    labelcounts = {}
    for featvec in dataset:
        currentlabel = featvec[-1]
        if currentlabel not in labelcounts.keys():
            labelcounts[currentlabel] = 0
        else:
            labelcounts[currentlabel] += 1
        shannonent = 0
        for key in labelcounts:
            prob = float(labelcounts[key]) / numentries  # 求出每个类别出现的概率
            shannonent -= prob * log(prob, 2)  # 计算香农熵
    return shannonent


# 按照给定特征划分数据集
# 分类算法除了需要测量信息熵，还需要划分数据集，度量划分数据集的熵，以便判断当前是否正确地划分了数据集。
# 我们将对每个特征划分数据集的结果计算一次信息熵，然后判断按照哪个特征划分数据集市最好的划分方法。
def splitdataset(dataset, axis, value):  # 该函数使用了三个输入参数：带划分的数据集、划分数据集的特征（数据集第几列）、需要返回的特征的值
    retdataset = []  # （按哪个值划分）。函数先选取数据集中第axis个特征值为value的数据，从这部分数据中去除第axis个特征，并返回。
    for featvec in dataset:
        if featvec[axis] == value:
            reducedfeatvec = featvec[:axis]
            reducedfeatvec.extend(featvec[axis + 1:])
            retdataset.append(reducedfeatvec)
    return retdataset


# 选择最好的数据集划分方式
# 遍历整个数据集，循环计算香农熵和splitDataSet()函数，找到最好的特征划分方式

def choosebestfeaturetosplit(dataset):
    numfeatures = len(dataset[0]) - 1  # dataset最后一列为分类
    baseentropy = calcshannonent(dataset)
    bestinfogain = 0
    bestfeature = -1
    for i in range(numfeatures):  # 循环每一列特征
        featlist = [example[i] for example in dataset]  # 创建一个新的列表
        uniquevals = set(featlist)  # 使用集合去重
        newentropy = 0
        for value in uniquevals:  # 循环第i列特征值
            subdataset = splitdataset(dataset, i, value)  # 划分数据集
            prob = len(subdataset) / float(len(dataset))  # 子数据集所占的比例
            newentropy += prob * calcshannonent(subdataset)
        infogain = baseentropy - newentropy
        if infogain > bestinfogain:
            bestinfogain = infogain
            bestfeature = i
    return bestfeature  # 返回按某列划分数据集的最大熵


# 递归构建决策树步骤：
# 1、得到原始数据集，
# 2、基于最好的属性值划分数据集，由于特征值可能多于两个，因此可能存在大于两个分支的数据集划分。
# 3、第一次划分之后，数据将被向下传递到树分支的下一个节点，在这个节点上，我们可以再次划分数据。我们可以采用递归的原则处理数据集。
# 4、递归结束的条件是，程序遍历完所有划分数据集的属性，或者每个分支下的所有实例都具有相同的分类。

def majoritycnt(classlist):  # 返回出现次数最多的类别
    classcount = {}
    for vote in classlist:
        if vote not in classcount.keys():
            classcount[vote] = 0
        else:
            classcount[vote] += 1
    sortedclasscount = sorted(classcount.items(), key=operator.itemgetter(1),
                              reverse=True)
    return sortedclasscount[0][0]


def createtree(dataset, labels):
    classlist = [example[-1] for example in dataset]  # 获取数据集的所有类别
    if classlist.count(classlist[0]) == len(classlist):  # 数据集的所有类别都相同，则不需要划分
        return classlist[0]
    if len(dataset[0]) == 1:
        return majoritycnt(classlist)  # 遍历完所有特征时返回出现次数最多的类别
    bestfeat = choosebestfeaturetosplit(dataset)
    bestfeatlable = labels[bestfeat]
    mytree = {bestfeatlable: {}}
    del labels[bestfeat]
    featvalues = [example[bestfeat] for example in dataset]
    uniquevals = set(featvalues)
    for value in uniquevals:
        sublabels = labels[:]
        mytree[bestfeatlable][value] = createtree(splitdataset(dataset, bestfeat,
                                                               value), sublabels)
    return mytree
