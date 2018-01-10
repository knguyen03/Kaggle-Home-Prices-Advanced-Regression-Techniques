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

columns = ["GrLivArea", "LotArea","1stFlrSF", "2ndFlrSF", "OverallQual",
           "OverallCond", "FullBath","HalfBath", "BedroomAbvGr", "TotRmsAbvGrd",
           "WoodDeckSF","OpenPorchSF","YearBuilt","YearRemodAdd"]
all_X = train[columns]
all_y = train['SalePrice'].reshape(1460,1)

#split the training data set so we can test

train_X, test_X, train_y, test_y = train_test_split(
        all_X, all_y, test_size=0.2,random_state=0)

#train model using the train portion of the training data set

lr = LinearRegression()
lr.fit(train_X,train_y)

#apply trained model to the test portion of the training data set

pred = lr.predict(train_X)

#output chart

plt.plot(train_y)
plt.plot(pred)
plt.legend(['data','prediction'])
plt.show()

#train model using all of training data set

realLr = LinearRegression()
realLr.fit(all_X,all_y)

#apply trained model to test data set

all_X_holdout = holdout[columns]
holdout_predictions = realLr.predict(all_X_holdout).ravel()

#write results to file

holdout_ids = holdout["Id"]
submission_df = { "Id": holdout_ids,"SalePrice": holdout_predictions}
submission = pd.DataFrame.from_dict(submission_df)
submission.to_csv('submission.csv',index=False)

    
    
