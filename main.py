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
from sklearn.model_selection import train_test_split

train = pd.read_csv("train.csv")
holdout = pd.read_csv("test.csv")

def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

train = create_dummies(train,"MSSubClass")
holdout = create_dummies(holdout,"MSSubClass")

lr = LinearRegression()

columns = ["GrLivArea", "LotArea","1stFlrSF", "2ndFlrSF", "OverallQual",
           "OverallCond", "FullBath","HalfBath", "BedroomAbvGr", "TotRmsAbvGrd",
           "WoodDeckSF","OpenPorchSF","YearBuilt","YearRemodAdd"]

all_X = train[columns]
all_y = train['SalePrice'].reshape(1460,1)

train_X, test_X, train_y, test_y = train_test_split(
        all_X, all_y, test_size=0.2,random_state=0)

lr.fit(train_X,train_y)

pred = lr.predict(train_X)

plt.plot(train_y)
plt.plot(pred)
plt.legend(['data','prediction'])
plt.show()

realLr = LinearRegression()
realLr.fit(all_X,all_y)

all_X_holdout = holdout[columns]

holdout_predictions = realLr.predict(all_X_holdout).ravel()
holdout_ids = holdout["Id"]
submission_df = { "Id": holdout_ids,"SalePrice": holdout_predictions}
#s = pd.Series(holdout_predictions,name='SalePrice')
#s.index.name = 'Id'

#s.reset_index
submission = pd.DataFrame.from_dict(submission_df)
submission.to_csv('submission.csv',index=False)

#plt.scatter(train_X,train_y)
#plt.scatter(train_X,pred)
#plt.legend(['data','fit line'])
#plt.show()

    
    
