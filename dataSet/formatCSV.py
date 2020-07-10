import numpy as np
import pandas as pd 

path = "F:/2020AI_SummerCamp/dataSet/"
dataFrame = pd.read_csv(path + "cartDS.csv")
for index, row in dataFrame.iterrows():
    dataFrame.iloc[index, 0] = row[0][4:]

dataFrame.to_csv(path + "cartDS2.csv")