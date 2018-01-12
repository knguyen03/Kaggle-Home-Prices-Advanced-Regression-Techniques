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
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split

#read data

train = pd.read_csv("train.csv")
holdout = pd.read_csv("test.csv")

#get rid of outliers

train = train[train.Id  != 186]
train = train[train.Id  != 524]
train = train[train.Id  != 1299]

#create log terms
train['OverallQual_log']=np.log(train['OverallQual'])
train['YearBuilt_log']=np.log(train['YearBuilt'])
holdout['OverallQual_log']=np.log(holdout['OverallQual'])
holdout['YearBuilt_log']=np.log(holdout['YearBuilt'])

#fill in some missing values

holdout['GarageArea'].fillna(holdout['GarageArea'].mean(),inplace=True)
holdout['TotalBsmtSF'].fillna(holdout['TotalBsmtSF'].mean(),inplace=True)

#function to create dummies

def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

#columns to create dummies for

cols = ['MSZoning','Condition1','Neighborhood','Street','LotConfig','BldgType','CentralAir']

#create dummies

for c in cols:
    train = create_dummies(train, c)
    holdout = create_dummies(holdout, c)

#coloumns to run regression on

columns = ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF','FullBath', 'YearBuilt',
           
           'MSZoning_C (all)', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM',
           
           'Neighborhood_Blmngtn', 'Neighborhood_Blueste', 'Neighborhood_BrDale',
           'Neighborhood_BrkSide', 'Neighborhood_ClearCr','Neighborhood_CollgCr', 
           'Neighborhood_Crawfor', 'Neighborhood_Edwards','Neighborhood_Gilbert', 
           'Neighborhood_IDOTRR', 'Neighborhood_MeadowV','Neighborhood_Mitchel', 
           'Neighborhood_NAmes', 'Neighborhood_NPkVill','Neighborhood_NWAmes',
           'Neighborhood_NoRidge', 'Neighborhood_NridgHt', 'Neighborhood_OldTown',
           'Neighborhood_SWISU', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW',
           'Neighborhood_Somerst', 'Neighborhood_StoneBr', 'Neighborhood_Timber',
           'Neighborhood_Veenker',
           
           'Condition1_Artery','Condition1_Feedr','Condition1_Norm',
           'Condition1_PosA','Condition1_PosN','Condition1_RRAe',
           'Condition1_RRAn','Condition1_RRNe','Condition1_RRNn',
           
           'Street_Grvl','Street_Pave', 
           
           'LotConfig_Corner','LotConfig_CulDSac','LotConfig_FR2', 
           'LotConfig_FR3','LotConfig_Inside',
           
           'BldgType_1Fam','BldgType_2fmCon','BldgType_Duplex','BldgType_Twnhs','BldgType_TwnhsE',
           'CentralAir_N','CentralAir_Y'
           ]

all_X = train[columns]
all_y = train['SalePrice']

#split the training data set so we can test

train_X, test_X, train_y, test_y = train_test_split(
        all_X, all_y, test_size=0.2,random_state=0)

#train model using the train portion of the training data set

lr = linear_model.Ridge(alpha=.5)
lr.fit(train_X,train_y)

#apply trained model to the test portion of the training data set

pred = lr.predict(train_X)

#train model using all of training data set

realLr = linear_model.Lasso()
realLr.fit(all_X,all_y)

#apply trained model to test data set

all_X_holdout = holdout[columns]
holdout_predictions = realLr.predict(all_X_holdout).ravel()

#write results to file

holdout_ids = holdout["Id"]
submission_df = { "Id": holdout_ids,"SalePrice": holdout_predictions}
submission = pd.DataFrame.from_dict(submission_df)
submission.to_csv('submission.csv',index=False)


