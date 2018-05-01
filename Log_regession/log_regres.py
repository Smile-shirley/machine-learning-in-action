import random

import numpy as np
import matplotlib.pyplot as plt


def creat_data():
    data_set = []
    data_labels = []
    with open('./testSet.txt') as file:
        content = file.readlines()
        for line in content:
            data_list = line.strip().split('\t')
            data_set.append([1.0, float(data_list[0]), float(data_list[1])])
            data_labels.append(int(data_list[2]))
    return data_set, data_labels


def sigmoid(inX):
    return 1 / (1 + np.exp(-inX))


def grad_ascent(data_set, data_labels):
    data_matrix = np.mat(data_set)
    labels_matrix = np.mat(data_labels).T
    m, n = data_matrix.shape
    alpha = .001
    max_cycle = 500
    weights = np.ones((n, 1))
    for i in range(max_cycle):
        h = sigmoid(data_matrix * weights)
        error = labels_matrix - h
        weights = weights + alpha * data_matrix.T * error
    return weights


def sto_grad_ascent(data_set, data_labels, num_iter=150):
    m, n = data_set.shape
    weights = np.ones(n)
    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4 / (1 + j + i) + .01
            random_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_set[random_index] * weights))
            error = data_labels[random_index] - h
            weights = weights + alpha * data_set[random_index] * error
            del (data_index[random_index])
    return weights


def plot_best_fit(weights):
    data_set, data_labels = creat_data()
    data_array = np.array(data_set)
    n = data_array.shape[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(data_labels[i]) == 1:
            # print(data_array[i,1],data_array[i,2])
            xcord1.append(data_array[i, 1])
            ycord1.append(data_array[i, 2])
        else:
            xcord2.append(data_array[i, 1])
            ycord2.append(data_array[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-4, 4, .1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    # print(x)
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def classify(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1
    else:
        return 0


def colic_test():
    train_list = []
    train_labels = []
    test_list = []
    test_labels = []
    with open('horseColicTraining.txt') as file_train:
        content_train = file_train.readlines()
    with open('horseColicTest.txt') as file_test:
        content_test = file_test.readlines()

    for line in content_train:
        line_content = line.strip().split('\t')
        train_list.append(line_content[:21])
        train_labels.append(float(line_content[-1]))
    train_weights = sto_grad_ascent(np.array(train_list, dtype=float), train_labels, 500)
    error_count = 0
    num_test_vec = 0

    for line2 in content_test:
        num_test_vec += 1
        line_content2 = line2.strip().split('\t')
        test_list.append(line_content2[:21])
        test_labels.append(float(line_content2[-1]))
        if int(classify(np.array(test_list, dtype=float), train_weights)) != test_labels:
            error_count += 1
        print('error_rate is %f' % (float(error_count) / num_test_vec))
    error_rate = float(error_count) / num_test_vec
    return error_rate


def multi_test():
    num_tests = 10
    error_sum = 0
    for i in range(num_tests):
        error_sum += colic_test()
    print("finally error_rate is %f" % (error_sum / float(num_tests)))
