import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from pandas import MultiIndex, Int16Dtype
import random
from sklearn.model_selection import train_test_split

import xgboost
import shap
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder    

# from category_encoders import OneHotEncoder

# from usefull_functions import *


def figSettings(fig,figname,config):

    # if config is None:#none
    #     import os
    #     settings = {'dir_plots':os.getcwd(), 'show':True, 'saveSVG':False}
    # else:pass
    format = 'png' if not config['saveSVG'] else 'svg'

    fig.savefig(config['dir_plots']+figname+'.'+format, format=format, bbox_inches="tight", pad_inches=0.2)

    if config['show']:
        plt.show()
    else:
        plt.close(fig)


def xg_classification_shap(df,labels,colors = None,config = None,figname='',
                               params = { 
                                        # 'eval_metric': 'mlogloss',
                                        'learning_rate':0.1,
                                        'n_estimators':1000,
                                        'max_depth':10,
                                        }
                           ):
    # drop points without label
    df = df.loc[labels.index]


    # encode labels to integer labels
    encoder =LabelEncoder()
    labels = Series(encoder.fit_transform(labels.copy()),index = labels.index)

    X_train, X_test, y_train, y_test = train_test_split(df.loc[df.index[1:1000]],labels.loc[df.index[1:1000]],
                                                        test_size=0.33, 
                                                        random_state=42, shuffle=True)


    model = xgboost.XGBClassifier(**params)#
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    accuracy = np.round(accuracy_score(y_test,y_test_pred),3)*100.0
   
    shap_values = shap.Explainer(model).shap_values(X_test)
    class_names = [f'{x}'for x in encoder.inverse_transform( model.classes_)]#cluster

    plotShap_class(shap_values,X_test,class_names,colors,accuracy,
                   figname=f'{figname}_class',config=config)


    for i,class_ in enumerate(class_names):       
        plotShap(shap_values[i],X_test,accuracy,colors[i],class_ = class_,
                       figname=f'{figname}_clustering_{class_}',config=config)
        
    return accuracy
    # return shap_values,X_test,class_names,colors,accuracy
def plotShap(shap_values,X_test,accuracy,color,class_ = None,
                   max_features_display=10,
                   figname='',config=None,plot_size = [10,8]): 


    fig,axs = plt.subplots(1,2)
    for ax in axs.ravel():
        ax.set_axis_off()
    # fig = plt.figure(figsize=(5,10))
    if class_ is not None:
        fig.suptitle(f'shap clustering {class_} (xgboost accuracy = {accuracy}%)')
        # shap_values = shap_values[class_]
        # colors = colors[class_]
    else:
        fig.suptitle(f'shap xgboost: ,accuracy = {accuracy}%')
        # shap_values = shap_values
    ax0 = fig.add_subplot(1,2,1) 
    # ax0.title.set_text(f'cluster {ind}')
    #add dim to shap vector
    shap_values1 = shap_values[np.newaxis,:] if len(shap_values.shape)==1 else shap_values
     
    shap.summary_plot(shap_values1, X_test, plot_type="bar", color = color,show=False,
                      max_display = max_features_display, plot_size=plot_size)
    
    if len(shap_values.shape)!=1:#cant plot binar classification
        ax1 = fig.add_subplot(1,2,2)
        # ax0.title.set_text(f'cluster {ind}')
        shap.summary_plot(shap_values, X_test,  show=False,
                        max_display = max_features_display, plot_size=plot_size)
    figSettings(fig,figname,config)




def plotShap_class(shap_values,X_test,class_names,colors,accuracy,config = None,figname ='',max_features_display=10): 

    fig,ax = plt.subplots(1,figsize = (10,10))
    fig.suptitle(f'shap xgboost: ,accuracy = {accuracy}%')

    shap.summary_plot(shap_values, X_test, class_inds="original", class_names=class_names,
                        max_display = max_features_display,show=False, )#cmap = colors,color=None,c =
    figSettings(fig,figname,config)


