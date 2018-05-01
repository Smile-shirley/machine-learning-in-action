from sklearn import neighbors  # 从机器学习库中调用KNN算法
from  sklearn import datasets  # 库中自带的数据集

# knn=neighbors.KNeighborsClassifier()
# iris=datasets.load_iris()
# print(iris)
# knn.fit(iris.data,iris.target)
# predictedlabel=knn.predict([0.1,0.2,0.3,0.4])
# print(predictedlabel)
import csv
import random
import math
import operator


# 打开文件
def loaddataset(filename, split, trainingset=[], testset=[]):
    with open(filename, 'r', encoding='utf-8') as csvfile:  # 一定注意编码的问题
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:  # random.random()生成0-1之间的随机浮点数
                trainingset.append(dataset[x])
            else:
                testset.append(dataset[x])


# 距离的测量
def euclideandistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


# 返回K个label

def getNeighbors(trainingset, testinstance, k):
    distances = []
    length = len(testinstance) - 1
    for x in range(len(trainingset)):
        dist = euclideandistance(testinstance, trainingset[x], length)
        distances.append((trainingset[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# 对类别进行计数

def getresponse(neighbors):
    classvotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classvotes:
            classvotes[response] += 1
        else:
            classvotes[response] = 1
    sortedvoters = sorted(classvotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedvoters[0][0]


# 计算准确率

def getaccuracy(testset, predictions):
    correct = 0
    for x in range(len(testset)):
        if testset[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testset))) * 100.0


def main():
    trainingset = []
    testset = []
    split = 0.67
    loaddataset(r'E:/机器学习深入与强化/irisdata.txt', split, trainingset, testset)
    print(trainingset)
    print(testset)
    predictions = []
    k = 3
    for x in range(len(testset)):
        neighbors = getNeighbors(trainingset, testset[x], k)
        result = getresponse(neighbors)
        # print(neighbors)
        # print(result)
        predictions.append(result)
    accuracy = getaccuracy(testset, predictions)
    print(accuracy)


main()
