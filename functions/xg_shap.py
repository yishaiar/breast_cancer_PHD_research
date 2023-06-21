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


def xg_classification_shap(df,labels,ind,colors,settings,figname,calc_shap,
                               params = { 
                                        'eval_metric': 'mlogloss',
                                        'learning_rate':0.1,
                                        'n_estimators':1000,
                                        'max_depth':10,
                                        }
                           ):

    labels = np.asarray(labels)[ind]
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
    model = xgboost.XGBClassifier(**params,use_label_encoder=False,num_class =  np.unique(labels).shape[0],)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    accuracy = np.round(accuracy_score(y_test,y_test_pred),3)*100.0
    if calc_shap:
        fig,ax= plt.subplots(1)
        ax.set_title(f'shap xgboost: accuracy = {accuracy}%')
        class_names = [f'cluster {x}'for x in model.classes_]
        
        shap_values = shap.TreeExplainer(model).shap_values(X_test)
        # shap.summary_plot(shap_values, X_test)
        shap.summary_plot(shap_values, X_test, class_inds="original", class_names=class_names,
                            cmap = colors,max_display =50)#,show=False
        
        dir,show,saveSVG = settings
        fig.savefig(dir+figname+'.png', format="png", bbox_inches="tight", pad_inches=0.2)
        if saveSVG:
            fig.savefig(dir+figname+'.svg', format="svg", bbox_inches="tight", pad_inches=0.2)
        if show:
            fig.show()
        else:
            fig.close()

    return accuracy
    
