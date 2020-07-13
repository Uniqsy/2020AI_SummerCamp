import numpy as np
import pandas as pd
import LoadFile


def simpleCrossValidation(rawData, rate=0.9):
    rawData = pd.DataFrame(rawData)
    m = int(rawData.shape[0] * rate)
    trainData = rawData.iloc[:m, :]
    testData = rawData.iloc[m:, :]
    return trainData, testData


def kFoldCrossValidation(rawData, processFunc, k):
    rawData = pd.DataFrame(rawData)
    n = np.shape(rawData)[0]
    step = n // k
    bestModel = 0
    leastErr = 0
    for i in range(k):
        if i == (k-1):
            testData = rawData.iloc[step*i: n, :]
            trainData = rawData.iloc[: step*i, :]
        else:
            testData = rawData.iloc[step*i: step*(i+1), :]
            trainData = rawData.iloc[: step*i, :]
            trainData = trainData.append(rawData.iloc[step*(i+1):])
        testData, testLabel = LoadFile.splitData(testData)
        trainData, trainLabel = LoadFile.splitData(trainData)
        model = getModel(trainData, trainLabel)
        err = testModel(model, testData, testLabel)
        if err < leastErr:
            leastErr = err
            bestModel = model
    return model
