import pandas as pd
import numpy as np


def loadCSV(filename):
    def converType(s):
        s = s.strip()
        try:
            return float(s) if '.' in s else int(s)
        except ValueError:
            return s

    rowData = np.loadtxt(filename, dtype='str', delimiter=',')
    rowData = rowData[1:, :]
    dataSet = [[converType(item) for item in data] for data in rowData]
    return dataSet

def splitData(rawData):
    labelSet = [data[-1] for data in rawData]
    dataSet = [data[:-1] for data in rawData]
    return dataSet, labelSet