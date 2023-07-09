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


def xg_classification_shap(df,labels,ind,colors,settings = None,figname='',calc_shap=True,
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

    # model = xgboost.XGBClassifier(
    #                                 num_class=np.unique(labels).shape[0],
    #                                 eval_metric='mlogloss',
    #                                 learning_rate=0.1,
    #                                 n_estimators=1000,
    #                                 max_depth=10,
    #                                 use_label_encoder=False,
    #                               )
    model = xgboost.XGBClassifier(**params,num_class =  np.unique(labels).shape[0],)#use_label_encoder=False
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    accuracy = np.round(accuracy_score(y_test,y_test_pred),3)*100.0
    if calc_shap:
        fig = plotShap(model,X_test,accuracy,colors)
        if settings is not None:
            dir,show,saveSVG = settings
        else: #none
            import os
            dir,show,saveSVG = os.getcwd(), True,False

        fig.savefig(dir+figname+'.png', format="png", bbox_inches="tight", pad_inches=0.2)
        if saveSVG:
            fig.savefig(dir+figname+'.svg', format="svg", bbox_inches="tight", pad_inches=0.2)
        if show:
            fig.show()
        else:
            fig.close()

    return accuracy
    
def plotShap(model,X_test,accuracy,colors): 
    shap_values = shap.Explainer(model).shap_values(X_test)
    class_names = [f'cluster {x}'for x in model.classes_]

    # fig,ax = plt.subplots(1,len(class_names))
    fig = plt.figure(figsize=(20,10))
    y = len(class_names)//3+1
    for i, class_ in enumerate(class_names):
       ax0 = fig.add_subplot(y,3,i+1)
    #ax0.title.set_text('Class 2 - Best ')
    # ax.set_title(f'shap xgboost: accuracy = {accuracy}%')
    
    
    
    # shap.summary_plot(shap_values, X_test)
    shap.summary_plot(shap_values, X_test, class_inds="original", class_names=class_names,
                        cmap = colors,max_display =50)#,show=False
    return fig