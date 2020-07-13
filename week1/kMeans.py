import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib



def loadTXT(filename):
    # 读入文件
    def converType(s):
        s = s.strip()
        try:
            return float(s) if '.' in s else int(s)
        except ValueError:
            return s

    rowData = np.loadtxt(filename, dtype='str')
    dataSet = [[converType(item) for item in data] for data in rowData]
    return dataSet

def calcL2Dist(x, y):
    # 计算两个向量间的L2距离
    x = np.array(x)
    y = np.array(y)
    powDist = ((x - y) ** 2).sum()
    sqDist = math.sqrt(powDist)
    return sqDist

def randCentrePoint(dataSet, k):
    # 随机生成k个初始聚类中心
    dataSet = pd.DataFrame(dataSet)

    # 获取最小值、最大值以及取值范围，生成在范围内的随机点
    minDf = dataSet.min()
    maxDf = dataSet.max()
    rangeDf = maxDf - minDf
    rangeMat = np.mat(rangeDf)
    kCentrePoints = np.random.rand(k, 1) * rangeMat + np.tile(minDf, (k, 1))
    kCentrePoints = np.mat(kCentrePoints)

    # 结果为k * m的一个矩阵，每一行表示一个随机中心
    return kCentrePoints

def kMeans(dataSet, k, distMeasureFunc = calcL2Dist, createCentralPoint = randCentrePoint):
    n, m = np.shape(dataSet)

    # kCentralPoints 为k * m的矩阵，保存生成的聚类中心
    kCentralPoints = createCentralPoint(dataSet, k)

    # clusterAssment 为n * 2的矩阵，第i行的向量的两个量，分别表示所属聚类的编号、到聚类中心的距离的平方
    clusterAssment = np.mat(np.zeros((n, 2)))

    # follower 字典，记录每个聚类中心包含的所有数据点
    follower = {}

    # lastResult 记录上一次更新聚类中心的结果，用于对比获得这次的更新幅度
    lastResult = 0

    # leastChangeRange 更新幅度的下限，为k * m的矩阵，目前下限为每个点的每个坐标移动均小于0.001
    leastChangeRange = 0.001 * np.ones((k, m))

    # meansVecChanged 用于记录聚类中心是否有更新，若无更新将停止循环
    meansVecChanged = True
    while meansVecChanged:
        meansVecChanged = False

        # n * k枚举每个数据点的最近的聚类中心并将其划分到新的聚类中心中
        for i in range(n):
            closestCentralPoint = 0
            minDist = float("inf")
            for j in range(k):
                nowDist = distMeasureFunc(dataSet[i], kCentralPoints[j])
                if nowDist < minDist:
                    closestCentralPoint = j
                    minDist = nowDist
            clusterAssment[i, :] = closestCentralPoint, minDist ** 2
            if closestCentralPoint not in follower:
                follower[closestCentralPoint] = []
            follower[closestCentralPoint].append(dataSet[i])

        # 重新计算本次分类后新的聚类中心
        for j in range(k):
            df = pd.DataFrame(follower[j])
            newMeans = df.mean()
            newMeans = np.array(newMeans)
            if type(lastResult) != int:
                if (lastResult - kCentralPoints < leastChangeRange).all():
                   continue
            else:
                kCentralPoints[j] = newMeans
                meansVecChanged = True

        # 转存上一次的计算结果
        lastResult = kCentralPoints.copy()

    return kCentralPoints, clusterAssment


def drawCluster(kCentralPoints, clusterAssmnt, dataSet, k):
    # 聚类结果画图，目前设置的上限是四个聚类，分别使用红、蓝、绿、黄颜色标记
    sampleNum, col = np.shape(dataSet)

    # 绘制数据点
    mark = ['or', 'ob', 'og', 'oy']
    for i in range(sampleNum):
        markIndex = int(clusterAssmnt[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    # 绘制聚类中心
    mark = ['+r', '+b', '+g', '+y']
    for i in range(k):
        plt.plot(kCentralPoints[i, 0], kCentralPoints[i, 1],
                 mark[i], markersize=12)

    # 输出绘图结果
    plt.show()


if __name__ == "__main__":
    # 读取文件
    # filePath = "F:/2020AI_SummerCamp/dataSet/"
    # rawData = loadTXT(filePath + "testSet.txt")

    # 硬编码输入，该数据为西瓜书，西瓜数据集4.0
    rawData = np.array([
    [0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
    [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
    [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.360, 0.370],
    [0.593, 0.042], [0.719, 0.103], [0.359, 0.188], [0.339, 0.241], [0.282, 0.257],
    [0.748, 0.232], [0.714, 0.346], [0.483, 0.312], [0.478, 0.437], [0.525, 0.369],
    [0.751, 0.489], [0.532, 0.472], [0.473, 0.376], [0.725, 0.445], [0.446, 0.459]])

    # 执行并画图
    k = 4
    kCentralPoints, clusterAssment = kMeans(rawData, k)
    drawCluster(kCentralPoints, clusterAssment, rawData, k)

    # print(central)