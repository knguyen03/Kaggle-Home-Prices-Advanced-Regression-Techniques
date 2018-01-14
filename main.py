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
from scipy import stats
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

#read data

train = pd.read_csv("train.csv")
holdout = pd.read_csv("test.csv")

#get rid of outliers

cols_outliers_id = [186,524,692,1183,1299]

for c in cols_outliers_id:
    train = train[train.Id != c]

#fill in some missing values

holdout['GarageArea'].fillna(holdout['GarageArea'].mean(),inplace=True)
holdout['TotalBsmtSF'].fillna(holdout['TotalBsmtSF'].mean(),inplace=True)

#function to create dummies

def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

#columns to create dummies for

cols_dummies = ['MSSubClass','MSZoning','Condition1','Neighborhood','Street',
                'LotConfig','BldgType','CentralAir','RoofStyle','Exterior1st',
                'ExterQual','Foundation','SaleType','SaleCondition']

#create dummies

for c in cols_dummies:
    train = create_dummies(train, c)
    holdout = create_dummies(holdout, c)
holdout['Exterior1st_ImStucc'] = 0
holdout['Exterior1st_Stone'] = 0

#coloumns to train model on

cols_model = ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF','FullBath',
              'YearBuilt', 'LotArea',
           
            'MSSubClass_20','MSSubClass_30', 'MSSubClass_40', 'MSSubClass_45', 
            'MSSubClass_50','MSSubClass_60', 'MSSubClass_70', 'MSSubClass_75', 
            'MSSubClass_80','MSSubClass_85', 'MSSubClass_90', 'MSSubClass_120', 
            'MSSubClass_160','MSSubClass_180', 'MSSubClass_190',
           
            'MSZoning_C (all)', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM',
           
            'Neighborhood_Blmngtn','Neighborhood_Blueste','Neighborhood_BrDale',
            'Neighborhood_BrkSide','Neighborhood_ClearCr','Neighborhood_CollgCr', 
            'Neighborhood_Crawfor','Neighborhood_Edwards','Neighborhood_Gilbert', 
            'Neighborhood_IDOTRR','Neighborhood_MeadowV','Neighborhood_Mitchel', 
            'Neighborhood_NAmes','Neighborhood_NPkVill','Neighborhood_NWAmes',
            'Neighborhood_NoRidge','Neighborhood_NridgHt','Neighborhood_OldTown',
            'Neighborhood_SWISU','Neighborhood_Sawyer','Neighborhood_SawyerW',
            'Neighborhood_Somerst','Neighborhood_StoneBr','Neighborhood_Timber',
            'Neighborhood_Veenker',
           
            'Condition1_Artery','Condition1_Feedr','Condition1_Norm','Condition1_PosA',
            'Condition1_PosN','Condition1_RRAe','Condition1_RRAn','Condition1_RRNe',
            'Condition1_RRNn',
           
            'Street_Grvl','Street_Pave', 
           
            'LotConfig_Corner','LotConfig_CulDSac','LotConfig_FR2','LotConfig_FR3',
            'LotConfig_Inside',
           
            'BldgType_1Fam','BldgType_2fmCon','BldgType_Duplex','BldgType_Twnhs',
            'BldgType_TwnhsE',
           
            'CentralAir_N','CentralAir_Y',
           
            'RoofStyle_Flat', 'RoofStyle_Gable', 'RoofStyle_Gambrel',
            'RoofStyle_Hip', 'RoofStyle_Mansard', 'RoofStyle_Shed',
           
            'Exterior1st_VinylSd', 'Exterior1st_MetalSd', 'Exterior1st_Wd Sdng',
            'Exterior1st_HdBoard', 'Exterior1st_BrkFace', 'Exterior1st_WdShing',
            'Exterior1st_CemntBd', 'Exterior1st_HdBoard', 'Exterior1st_ImStucc',
            'Exterior1st_MetalSd', 'Exterior1st_Plywood', 'Exterior1st_Stone',
            'Exterior1st_Stucco', 'Exterior1st_VinylSd', 'Exterior1st_Wd Sdng',
            'Exterior1st_WdShing',
           
            'ExterQual_Ex', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA',
           
            'Foundation_BrkTil', 'Foundation_CBlock', 'Foundation_PConc',
            'Foundation_Slab', 'Foundation_Stone', 'Foundation_Wood',
           
            'SaleType_COD','SaleType_CWD','SaleType_Con','SaleType_ConLD','SaleType_ConLI',
            'SaleType_ConLw','SaleType_New','SaleType_Oth','SaleType_WD',
           
            'SaleCondition_Abnorml','SaleCondition_AdjLand','SaleCondition_Alloca',
            'SaleCondition_Family','SaleCondition_Normal','SaleCondition_Partial'
            ]

#transform skewed variables
#train['SalePrice_BoxCox'],_ = stats.boxcox(train['SalePrice'])
#train['LotArea_BoxCox'],_ = stats.boxcox(train['LotArea'])

#define dependent and independent variables in train

all_X = train[cols_model]
all_y = train['SalePrice']

#split train data set

train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2,random_state=0)

#gradient boosting on part of train for testing

gr = GradientBoostingRegressor(n_estimators=400,max_depth=5,min_samples_split=2,
                               learning_rate = 0.5,loss='ls')
gr.fit(train_X,train_y)

#gradient boosting on all of train

realGr = GradientBoostingRegressor(n_estimators=500,max_depth=5,min_samples_split=2,
                               learning_rate = 0.5,loss='ls')
realGr.fit(all_X,all_y)

#apply trained model to test data set

all_X_holdout = holdout[cols_model]
holdout_predictions = realGr.predict(all_X_holdout).ravel()

#write results to file

holdout_ids = holdout["Id"]
submission_df = { "Id": holdout_ids,"SalePrice": holdout_predictions}
submission = pd.DataFrame.from_dict(submission_df)
submission.to_csv('submission.csv',index=False)


