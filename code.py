# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 22:45:13 2019

@author: The Freaky Gamer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("train.csv")

dataset = dataset.drop(['ID'], axis=1)

X = dataset.iloc[:, :].values
df = pd.DataFrame(X)

'''import seaborn as sns
for i in range(13):
    sns.boxplot(df[i])
    plt.figure()'''
    
df = df.drop([0, 1, 3, 5, 11], axis=1)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor()

reg.fit(X, y)

dataset_test = pd.read_csv("test.csv")
dataset_test = dataset_test.drop(['ID'], axis = 1)
X_test = dataset_test.iloc[:, :].values
df_test = pd.DataFrame(X_test)
df_test = df_test.drop([0, 1, 3, 5, 11], axis=1)
X_test = df_test.iloc[:, :].values
y = reg.predict(X_test)

df_out = pd.DataFrame({'ID':dataset_test['ID'], 'medv':y})
df_out.to_csv("out4.csv")