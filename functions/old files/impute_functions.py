import pandas as pd
from pandas import MultiIndex, Int16Dtype
import xgboost as xgb
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

def getZerosMat(k,columns,percent = 10):
    zeros = pd.DataFrame(index=list(k.keys()),columns = columns)

    for i,K in k.items():
        # n = names[i]['NamesAll'].copy()
        # for f in ['CD45','H3','H3.3','H4']:
        #     n.remove(f)
        for f in columns:
            # print(f'{f}: {len(K[K[f]==0])/len(K)*100}')
            try:
                percentage = len(K[K[f]==0])/len(K)*100
                if percentage>=percent:
                    zeros.at[i,f] = percentage
            except:
                pass
    return zeros

def MimalzerosOverlap(K,namesAll,n):
    m = namesAll.copy()
    # n = namesAll.copy()
    # for f in ['H3','CD45','H4' ,'H3.3']:
    #     n.remove(f)
    nonzeros = pd.DataFrame(index=m,columns = n)

    for f in n:
        for j in m:
            # overlap percentage
            percent = len(K[(K[f]==0)*(K[j]==0)])/len(K)*100
            # find features with minimal percentage
            if f != j and percent<6:
                nonzeros.at[j,f] = percent
    return nonzeros

import itertools
def createDataset(nonzeros,K,feature):
    Xfeatures = nonzeros [feature]
    Xfeatures = Xfeatures[Xfeatures.notnull()].index.tolist()
    AllFeatures=Xfeatures.copy();    AllFeatures.append(feature)
    Xy = K[AllFeatures]
    # remove zeros for train
    ind_nonzero = (Xy[AllFeatures] != 0).all(axis=1)
    # ind_nonzero = (Xy[feature] != 0)
    Xy_train = Xy.loc[ind_nonzero].reset_index()
    x_train = Xy_train[Xfeatures]
    y_train = Xy_train[feature]




    ind_zero = (Xy[feature] == 0)
    Xy_test = Xy.loc[ind_zero]
    
    x_test = Xy_test[Xfeatures]
    # y_test = Xy_test[feature]
    return x_train, y_train , x_test

def test_val_split(x_train, y_train):
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                      train_size=0.9, random_state=42)
        y_val = y_val.to_frame()
        y_train = y_train.to_frame()
        return x_train, x_val, y_train, y_val
# def iter():
#     colsample_bytree = [0.3,0.6,0.8,1]
#     colsample_bylevel = [0.3,0.6,0.8,1]
#     return itertools.product (colsample_bytree,colsample_bylevel)
def xg(nonzeros,K,feature):
    x_train, y_train , x_test =  createDataset(nonzeros,K,feature)
    x_train, x_val, y_train, y_val = test_val_split(x_train, y_train)
    y_train = np.log(y_train)
    # Define the XGBoost model
    xgb_model = xgb.XGBRegressor(objective = "reg:squarederror", n_estimators=1000, max_depth=10, learning_rate=0.01,
                                )

    # Fit the model to the training data
    xgb_model.fit(x_train, y_train)

    # Make predictions on the validation data and exponent (the  prediction is on the log(y))
    y_val_pred = np.exp(xgb_model.predict(x_val))
    y_test_pred = np.exp(xgb_model.predict(x_test))
    y_val_pred = pd.DataFrame(y_val_pred, index = x_val.index,columns =[feature])
    y_test_pred = pd.DataFrame(y_test_pred, index = x_test.index,columns =[feature])


    K[feature][x_test.index] = y_test_pred[feature]
    # predictions[predictions<0] = 0
    mse, mae, accuracy = evaluate(y_val, y_val_pred,feature)
    return K,mse, mae, accuracy
    # return mse, mae
    # c=pd.DataFrame()
    # c['y_val'] = y_val
    # c['y_val_pred'] = predictions
    # print(c)
def evaluate(y_val, y_val_pred,feature):
    # Evaluate the model's performance
    mean = np.mean(y_val)
    mse = mean_squared_error(y_val, y_val_pred)/mean
    mae = mean_absolute_error(y_val, y_val_pred)/mean

    errors = abs(y_val_pred[feature] - y_val[feature])
    mape = 100 * np.mean(errors / y_val[feature])
    accuracy = 100 - mape
    # print(f"{feature} - Mse:{mse}, mae {mae}")
    return mse, mae, accuracy




# from numpy import loadtxt
# import xgboost
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
# # load data
# dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# # split data into X and y
# X = dataset[:,0:8]
# Y = dataset[:,8]
# # CV model
# model = xgboost.XGBClassifier()
# kfold = KFold(n_splits=10, random_state=7)
# results = cross_val_score(model, X, Y, cv=kfold)
# print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))