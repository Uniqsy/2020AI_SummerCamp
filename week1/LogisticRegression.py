import numpy as np
import pandas as pd
import LoadFile


def sigmoid(x):
    # 计算sigmoid函数
    return 1.0 / (1 + np.exp(-x))


def standardize(dataSet):
    # 数据标准化
    resultSet = dataSet.copy()
    meanSet = np.mean(dataSet, axis=0)
    stdSet = np.std(dataSet, axis=0)
    resultSet = (resultSet - meanSet) / stdSet
    return resultSet


def getStandardedMatData(rawData):
    # 将数据标准化后，再分割为数据集、标签集，然后将格式转换为矩阵
    dataSet, labelSet = LoadFile.splitData(rawData)
    dataSet = np.mat(dataSet)
    labelSet = np.mat(labelSet).T
    dataSet = standardize(dataSet)
    return dataSet, labelSet


def batchLogisticRegression(rawData, alpha=0.001, maxCycles=500):
    # 批量梯度下降法进行回归
    dataSet, labelSet = getStandardedMatData(rawData)
    n, m = np.shape(dataSet)
    weights = np.zeros((m, 1))
    for i in range(maxCycles):
        # 使用所有的数据来更新weights
        grad = dataSet.T * (sigmoid(dataSet * weights) - labelSet) / n
        weights = weights - alpha * grad
    return weights


def randomLogisticRegression(rawData, alpha=0.001, maxCycles=500):
    # 随机梯度下降法进行回归
    rawData = pd.DataFrame(rawData)

    # sampleSet 用于保存随机抽样得到的数据
    sampleSet = rawData.sample(n=maxCycles)
    dataSet, labelSet = getStandardedMatData(sampleSet)

    n, m = np.shape(dataSet)
    weights = np.zeros((m, 1))
    for i in range(n):
        # 逐个使用随机抽样得到的数据，更新weights
        grad = dataSet[i].T * (sigmoid(dataSet[i] * weights) - labelSet[i])
        weights = weights - alpha * grad
    return weights


def newtonLogisticRegression(rawData, alpha=0.001, maxCycles=500):
    rawData = pd.DataFrame(rawData)
    dataSet, labelSet = getStandardedMatData(rawData)
    n, m = np.shape(dataSet)
    weights = np.zeros((m, 1))

    for i in range(maxCycles):
        px = sigmoid(dataSet * weights)
        grad = dataSet.T * (sigmoid(dataSet * weights) - labelSet) / n
        Hessian = (dataSet.T * np.diag(px.getA1()) *
                   np.diag((1-px).getA1()) * dataSet) / n + 0.001
        weights = weights - Hessian.I * grad
    return weights


def calcAccuracy(rawData, lrFunc, alpha=0.001, maxCycles=500):
    # 计算回归的准确率
    dataSet, labelSet = getStandardedMatData(rawData)
    dataSize = np.shape(dataSet)[0]

    weights = lrFunc(rawData, alpha, maxCycles)
    px = sigmoid(dataSet * weights)

    # 统计在训练集上，训练得到的回归函数有多少
    trainErrCnt = 0
    for i, pxi in enumerate(px):
        if pxi < 0.5 and labelSet[i] == 1:
            trainErrCnt += 1
        elif pxi >= 0.5 and labelSet[i] == 0:
            trainErrCnt += 1
    trainAccuracy = 1 - trainErrCnt / dataSize
    return trainAccuracy


if __name__ == "__main__":
    filePath = "F:/2020AI_SummerCamp/dataSet/"
    rawData = LoadFile.loadCSV(filePath + "diabetesN.csv")
    # rawData = pd.read_table(filePath + 'testSet.txt', header=None)

    print(calcAccuracy(rawData, newtonLogisticRegression))
