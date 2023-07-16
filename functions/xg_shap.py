import pandas as pd
import numpy as np
from pandas import MultiIndex, Int16Dtype
import random
from sklearn.model_selection import train_test_split

import xgboost
import shap
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from usefull_functions import *


def xg_classification_shap(df,labels,ind,colors,settings = None,figname='',
                               params = { 
                                        'eval_metric': 'mlogloss',
                                        'learning_rate':0.1,
                                        'n_estimators':1000,
                                        'max_depth':10,
                                        }
                           ):


    labels = np.asarray(labels).astype(int)[ind]
    df = df.copy().reset_index(drop = True).loc[ind]



    X_train, X_test, y_train, y_test = train_test_split(df,labels,
                                                        test_size=0.33, 
                                                        random_state=42)


    model = xgboost.XGBClassifier(**params,num_class =  np.unique(labels).shape[0],)#use_label_encoder=False
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    accuracy = np.round(accuracy_score(y_test,y_test_pred),3)*100.0
   
    shap_values = shap.Explainer(model).shap_values(X_test)
    class_names = [f'cluster {x}'for x in model.classes_]

    plotShap_class(shap_values,X_test,class_names,colors,accuracy,
                   figname=f'{figname}_class',settings=settings)


    for i in range (len(class_names)):       
        plotShap(shap_values,X_test,accuracy,colors,ind = i,
                       figname=f'{figname}_clust_{i}',settings=settings)
        
    return accuracy
    # return shap_values,X_test,class_names,colors,accuracy
def plotShap(shap_values,X_test,accuracy,colors,ind = None,
                   max_features_display=10,
                   figname='',settings=None,plot_size = [10,8]): 


    fig,axs = plt.subplots(1,2)
    for ax in axs.ravel():
        ax.set_axis_off()
    # fig = plt.figure(figsize=(5,10))
    if ind is not None:
        fig.suptitle(f'shap cluster {ind} (xgboost accuracy = {accuracy}%)')
        shap_values = shap_values[ind]
        colors = colors[ind]
    else:
        fig.suptitle(f'shap xgboost: ,accuracy = {accuracy}%')
        # shap_values = shap_values
    ax0 = fig.add_subplot(1,2,1)
    # ax0.title.set_text(f'cluster {ind}')
    shap.summary_plot(shap_values, X_test, plot_type="bar", color = colors,show=False,
                      max_display = max_features_display, plot_size=plot_size)
    
    ax1 = fig.add_subplot(1,2,2)
    # ax0.title.set_text(f'cluster {ind}')
    shap.summary_plot(shap_values, X_test,  show=False,
                      max_display = max_features_display, plot_size=plot_size)
    figSettings(fig,figname,settings)




def plotShap_class(shap_values,X_test,class_names,colors,accuracy,settings = None,figname ='',max_features_display=10): 

    fig,ax = plt.subplots(1,figsize = (10,10))
    fig.suptitle(f'shap xgboost: ,accuracy = {accuracy}%')

    shap.summary_plot(shap_values, X_test, class_inds="original", class_names=class_names,
                        cmap = colors,max_display = max_features_display,show=False)#
    figSettings(fig,figname,settings)
