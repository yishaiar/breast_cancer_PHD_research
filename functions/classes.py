import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
# remove avx warning:
# I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
from os.path import isfile
from parent_class import * 
from pandas import DataFrame,Series
from numpy import ndarray
from matplotlib.pyplot import figure

# from typing import List
from numpy import asarray, arange 


from pandas import read_csv
from umap import UMAP

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import  DBSCAN




from plot_functions_new import cluster_colors,sample_colors,backgroundColor,hex_to_rgba





class Classification(Parent):
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)#import keys and values from dictionary to class
        # self.get_attribute()#print all attributes
        # self.calc_db = False
        self.folderExists(self.dir_plots)#verify that plots folder exists
    
    def get_classes(self,samples,sample_labels,dict):
        classes = Series(index = samples.index)#classes initializes as empty
           
        for sampNum in sorted(samples.unique()):# don't drop anchor samples - can be used for validation
        
            single_sample_labels = sample_labels.loc[samples [samples== sampNum].index].copy()#clusters of specific sample
            single_sample_classes = Series(index = single_sample_labels.index)# get the classes of the clusters - according to the dictionary
            for label in single_sample_labels.unique():
                try:
                    classification = dict[(str(float(sampNum)),str(float(label)))]
                except:
                    print(f'error in sample {sampNum} and label {label}')
                    classification = 'classification_error'
                # print(classification)
                single_sample_classes.loc[single_sample_labels[single_sample_labels == label].index] = classification
                
                
            classes.loc[single_sample_labels.index] = single_sample_classes.copy()#add to classes
        return classes
    def classes_dict(self,csv_add,):
   

        df = read_csv(csv_add,sep =',',comment='#',index_col = None).dropna(axis=1, how='all').dropna(axis=0, how='all').astype(str).replace(' ', '', regex = True)
         
        # df = pd.read_csv(add,comment='#').dropna(axis=1, how='all').dropna(axis=0, how='all')

        # for each sample read row of total clusters into dict -  and drop row from df 
        df['class'] = df['class'].str.capitalize()
        total = df[df['class']=='Total']#['clusters']
        rest = df[df['clusters']=='rest']#['class']

        df = df.drop(total.index).drop(rest.index)
       

        classes_dict = {}
        for sampNum in df['samp'].unique():#iterate on all samples
            total_clusters = Series(arange( 1+int(total[total['samp']==sampNum]['clusters'].values[0]))) # all clusters in sample
            for ind in df[df['samp']==sampNum].index:#iterate on all classes of sample
                class_clusters = asarray(df.loc[ind]['clusters'].split(';')).astype(int)
                total_clusters = total_clusters[~total_clusters.isin(class_clusters)]#remove clusters that are in class from total clusters- they are added to dict
                for clustNum in class_clusters:  
                    # labels data saved as str(float)
                    classes_dict[(str(float(sampNum)),str(float(clustNum)))] = df.loc[ind]['class']
            #add rest clusters
            for clustNum in total_clusters: 
                classes_dict[(str(float(sampNum)),str(float(clustNum)))] = rest[rest['samp']==sampNum]['class'].values[0]
            #add noise classification -1
            clustNum = -1
            classes_dict[(str(float(sampNum)),str(float(clustNum)))] = 'Noise'
        return classes_dict      
