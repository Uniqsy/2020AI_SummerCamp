import numpy as np
import pandas as pd
import math
import LoadFile

def solveMissingData(dataSet):
    n, m = np.shape(dataSet)
    for x in range(n):
        if m == cntZero(dataSet, x):
            continue
        closestPoint = 0
        distMin = float("inf")
        for y in range(n):
            if cntZero(dataSet, y) > 0:
                continue
            if x == y:
                continue
            distNow = calcDist(dataSet, x, y)
            if distNow < distMin:
                distMin = distNow
                closestPoint = y
        for i in range(m):
            if dataSet[x][i] == 0:
                dataSet[x][i] = dataSet[closestPoint][i]
                print(closestPoint, i, dataSet[closestPoint][i])
    return dataSet


def cntZero(dataSet, i):
    # 计算dataSet中第i行的空值数量，可以变为nan
    # 如果是nan的话，可以用以下代码代替
    # dataSet.iloc[j].isnull().sum()
    cnt = 0
    m = np.shape(dataSet)[1]
    for feat in range(m):
        if dataSet[i][feat] == 0:
            cnt += 1
    return cnt

def calcDist(dataSet, x, y):
    # 计算dataSet中x号数据与y号数据之间的欧氏距离
    n, m = np.shape(dataSet)
    z = 0
    for feat in range(m):
        if dataSet[x][feat] == 0 or dataSet[y][feat] == 0:
            continue
        z += (dataSet[x][feat] - dataSet[y][feat]) ** 2
    return math.sqrt(z)

def minmaxNormalize(dataSet):
    # 数据映射到[0, 1]
    dataSet = pd.DataFrame(dataSet)
    minDf = dataSet.min()
    maxDf = dataSet.max()
    normalizedSet = (dataSet - minDf) / (maxDf - minDf)
    return normalizedSet

def zscoreStanderize(dataSet):
    # 用z-score方法进行数据标准化
    dataSet = pd.DataFrame(dataSet)
    meanDf = dataSet.mean()
    stdDf = dataSet.std()
    standerizedSet = (dataSet - meanDf) / stdDf
    return standerizedSet


if __name__ == "__main__":
    filePath = "F:/2020AI_SummerCamp/dataSet/"
    # rawData = LoadFile.loadCSV(filePath + "Pima.csv")
    rawData = LoadFile.loadCSV(filePath + "diabetesN.csv")
    dataSet, labelSet = LoadFile.splitData(rawData)
    # dataSet = solveMissingData(dataSet)
    # dataSet = minmaxNormalize(dataSet)
    dataSet = zscoreStanderize(dataSet)
    print(dataSet)
