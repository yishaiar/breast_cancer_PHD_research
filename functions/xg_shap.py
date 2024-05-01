# import pandas as pd
from random import shuffle
from numpy import newaxis
import numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelEncoder 
# # from category_encoders import OneHotEncoder
import shap
from xgboost import XGBClassifier

from parent_class import * 




def fitXGBClassifier(X_train, y_train,xg_params={'learning_rate':0.1,'n_estimators':1000,'max_depth':10,}):
    '''
    encode labels to integer labels by fitting label encoder
    fit xgboost classification model using encoded labels and X features
    return fitted classification model and fitted label encoder (allows encoding and decoding labels for inference)
    '''
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    
    XGBClassifier_model = XGBClassifier(**xg_params)#
    XGBClassifier_model.fit(X_train, y_train)
    
    return XGBClassifier_model,label_encoder
    
    
class Shap(Parent):
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)#import keys and values from dictionary to class
        # self.get_attribute()#print all attributes
        # self.calc_db = False
        self.folderExists(self.dir_plots)#verify that plots folder exists

   





    def calculate(self,X_test,model,label_encoder,accuracy,colors,LEN = 1000,figname = ''):
        '''
        calculate shap values for each predicted class:
        xgboost classifier was fitted on the full train dataset (X_train,y_train)
        explainer is fitted on test dataset/fitted model
        shap calculated on smaller subset of test data (X_test) to save time
        
        '''
        print (f'Calculating shap values for {LEN} samples')
        self.colors = colors
        self.accuracy = accuracy
        
        # -------------------------
        # explainer options: 
        # 1. XGBClassifier model
        self.explainer = shap.Explainer(model)
        # 2. using model y predictions (model type isnt important) - currently not working
        # self.explainer = shap.Explainer(model.predict(self.X_test), self.X_test) 
        # 3. deep learning
        # The background dataset to use for integrating out features. Deep integrates over these samples. size should only be like 100 or 1000 random background samples,
        # background = x_train[np.random.choice(x_train.shape[0], 1000, replace=False)]
        # # DeepExplainer to explain predictions of the model
        # explainer = shap.DeepExplainer(model, background)
        # ------------------------------
        self.X_test = X_test.loc[X_test.index[:LEN]]
        self.shap_values = self.explainer.shap_values(self.X_test)
        self.class_names = [f'{x}' for x in label_encoder.inverse_transform( model.classes_)]#cluster
        # print(class_names)
        # print(len(shap_values),len(shap_values),len(colors))
        
        # ------------------------------------------------
        # plot shape values for each class
        
        self.plotShap_class(self.shap_values,self.X_test,self.class_names,self.colors,self.accuracy,
                       figname=figname)


        for i,class_ in enumerate(self.class_names):       
            self.plotShap(self.shap_values[i],self.X_test,self.accuracy,self.colors[i],class_ = class_,
                        figname=f'{figname}by_{class_}')
            
    
    # return shap_values,X_test,class_names,colors,accuracy
    def plotShap(self,shap_values,X_test,accuracy,color,class_ = None,
                    max_features_display=10,
                    figname='',plot_size = [10,8]): 


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
        shap_values1 = shap_values[newaxis,:] if len(shap_values.shape)==1 else shap_values
        
        shap.summary_plot(shap_values1, self.X_test, plot_type="bar", color = color,show=False,
                        max_display = max_features_display, plot_size=plot_size)
        
        if len(shap_values.shape)!=1:#cant plot binar classification
            ax1 = fig.add_subplot(1,2,2)
            # ax0.title.set_text(f'cluster {ind}')
            shap.summary_plot(shap_values, X_test,  show=False,
                            max_display = max_features_display, plot_size=plot_size)
        
        self.figSettings(fig,figname)


    def plotShap_class(self,shap_values,X_test,class_names,colors,accuracy,figname ='',max_features_display=10): 

        fig,ax = plt.subplots(1,figsize = (10,10))
        fig.suptitle(f'shap xgboost: ,accuracy = {accuracy}%')

        shap.summary_plot(shap_values, X_test, class_inds="original", class_names=class_names,
                            max_display = max_features_display,show=False, )#cmap = colors,color=None,c =
        
        self.figSettings(fig,figname)


# def figSettings(fig,figname,config):

#     # if config is None:#none
#     #     import os
#     #     settings = {'dir_plots':os.getcwd(), 'show':True, 'saveSVG':False}
#     # else:pass
#     format = 'png' if not config['saveSVG'] else 'svg'

#     fig.savefig(config['dir_plots']+figname+'.'+format, format=format, bbox_inches="tight", pad_inches=0.2)

#     if config['show']:
#         plt.show()
#     else:
#         plt.close(fig)


def predict_with_thresh_proba(data,model ,label_encoder = None,thresh_proba = 0.99,):
    
    proba,preds  = model.predict_proba(data), model.predict(data)
    preds[proba[np.arange(len(proba)),preds]<thresh_proba] = len(np.unique(preds))#set all values that are not classified with high thresh to new class not in the original classes
    if label_encoder is None:
        return preds#don't inverse transform labels
    labels = label_encoder.classes_
    label_encoder.classes_ = np.append(labels, 'Noise')#add the new class to the label encoder - unclassified stays noise
    preds =  label_encoder.inverse_transform(preds)
    label_encoder.classes_ = labels#return the original classes to the label encoder
    return preds