# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:11:00 2018

@author:
    Khang Nguyen
    Nils Lehmann
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

train = create_dummies(train,"MSSubClass")
test = create_dummies(test,"MSSubClass")

lr = LinearRegression()

train_x = train["GrLivArea"].reshape(1460,1)
train_y = train["SalePrice"].reshape(1460,1)

lr.fit(train_x,train_y)

pred = lr.predict(train_x)

plt.plot(train["SalePrice"])
plt.plot(pred)
plt.legend(['data','prediction'])
plt.show()

    
    
