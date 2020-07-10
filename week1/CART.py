import numpy as np
import pandas as pd
import operator
import LoadFile


# 计算当前labelSet的Gini函数
def calcGini(labelSet):
    labelsCnt = calcClass(labelSet)
    n = np.shape(labelSet)[0]
    gini = 1
    for label in labelsCnt.keys():
        gini -= float(labelsCnt[label] / n) ** 2
    return gini


# 计算一个labelSet中每种label的个数，返回一个字典
def calcClass(labelSet):
    results = {}
    for label in labelSet:
        label = str(label)
        results[label] = results.get(label, 0) + 1
    return results

# 根据数据集和标签集建树


def buildTree(dataSet, labeSet):
    bestFeat, bestVal, leftData, leftLabel, rightData, rightLabel = chooseBestSplit(
        dataSet, labeSet)
    if bestFeat == None:
        tree = {
            "val": bestVal, "gini": calcGini(labelSet), "sum": np.shape(dataSet)[0]
        }
        return tree
    tree = {"feat": bestFeat, "val": bestVal,
            "gini": calcGini(labelSet), "sum": np.shape(dataSet)[0],
            "left": buildTree(leftData, leftLabel),
            "right": buildTree(rightData, rightLabel)}
    return tree


def classify(tree, testData):
    # 当找到叶子节点的时候
    if "feat" not in tree:
        return tree["val"]

    # 当找到非叶子节点的时候，继续根据条件递归
    feat = tree["feat"]
    val = tree["val"]
    if testData[feat] < val:
        targetTree = tree['left']
    else:
        targetTree = tree["right"]
    return classify(targetTree, testData)

# 枚举寻找最佳切分点


def chooseBestSplit(dataSet, labelSet):
    dataSet = np.array(dataSet)

    # 停止递归的条件
    lowerLimit = {
        # 收益的下限
        "gain": 0.001,
        # 数据规模的下限
        "size": 8
    }
    n, m = np.shape(dataSet)

    # 若当前标签集中的标签全部相同，则停止递归
    labelsCnt = calcClass(labelSet)
    if len(labelSet) == labelsCnt[labelSet[0]]:
        return None, labelSet[0], None, None, None, None

    # 当前基础Gini与未来计算得到的最好Gini
    baseGini = calcGini(labelSet)
    bestGini = float("inf")

    # 用于返回的变量
    bestFeat = 0.0  # 最佳切分点所在列的列号
    bestVal = 0.0  # 最佳切分点的取值
    bestLeftData = 0  # 最佳切分后得到的左子树的数据集
    bestLeftLabel = 0  # 最佳切分后得到的左子树的标签集
    bestRightData = 0
    bestRightLabel = 0
    for feat in range(m):
        # 尝试每一列
        tmpDataSet = dataSet[:, feat]
        uniqueVal = set(tmpDataSet)
        sortedUniqueVal = list(sorted(uniqueVal))
        sizeOfVal = len(sortedUniqueVal)
        for i in range(sizeOfVal - 1):
            # 尝试每一列的每一个切分点取值
            splitVal = (sortedUniqueVal[i] + sortedUniqueVal[i+1]) / 2
            leftData, leftLabel, rightData, rightLabel = splitToFeat(
                dataSet, labelSet, feat, splitVal)
            leftProb = float(leftData.shape[0] / n)
            rightProb = float(rightData.shape[0] / n)
            nowGini = leftProb * \
                calcGini(leftLabel) + rightProb * calcGini(rightLabel)
            if nowGini < bestGini:
                bestGini, bestFeat, bestVal = nowGini, feat, splitVal
                bestLeftData, bestLeftLabel = leftData, leftLabel
                bestRightData, bestRightLabel = rightData, rightLabel

    # 当收益过小时停止划分
    if baseGini - bestGini < lowerLimit["gain"]:
        return None, calcLeaves(labelSet), None, None, None, None
    # 当集合过小时停止划分
    if (bestLeftData.shape[0] < lowerLimit["size"]) and (bestRightData.shape[0] < lowerLimit["size"]):
        return None, calcLeaves(labelSet), None, None, None, None
    return bestFeat, bestVal, bestLeftData, bestLeftLabel, bestRightData, bestRightLabel


# 多数投票，生成叶子节点的类别信息
def calcLeaves(labelSet):
    labelsCnt = calcClass(labelSet)
    sortResult = sorted(labelsCnt.items(),
                        key=operator.itemgetter(1), reverse=True)
    return sortResult[0][0]


# 根据列表生成式生成切分后的数据集
def splitToFeat(dataSet, labelSet, feat, val):
    dataSet = np.array(dataSet)
    labelSet = np.array(labelSet)
    leftData = dataSet[np.nonzero(dataSet[:, feat] < val)[0]]
    leftLabel = labelSet[np.nonzero(dataSet[:, feat] < val)[0]]
    rightData = dataSet[np.nonzero(dataSet[:, feat] >= val)[0]]
    rightLabel = labelSet[np.nonzero(dataSet[:, feat] >= val)[0]]
    return leftData, leftLabel, rightData, rightLabel


if __name__ == '__main__':
    # 读入文件
    # 使用的是网上找到的一个数据集
    filePath = "F:/2020AI_SummerCamp/dataSet/"
    rawData = LoadFile.loadCSV(filePath + "cartDS.csv")

    # 预处理
    dataSet, labelSet = LoadFile.splitData(rawData)
    tree = buildTree(dataSet, labelSet)

    # 测试数据
    testVec = np.array([7, 3.2, 4.7, 1.4])
    print(classify(tree, testVec))
