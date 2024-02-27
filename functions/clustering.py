
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


import numpy as np
from pandas import read_csv
from umap import UMAP

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import  DBSCAN


from plot_functions_new import cluster_colors,sample_colors,backgroundColor,hex_to_rgba



class Clustering(Parent):
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)#import keys and values from dictionary to class
        # self.get_attribute()#print all attributes
        # self.calc_db = False
        self.folderExists(self.dir_plots)#verify that plots folder exists
    def read_csv(self,algorithm,samp,features,fname = '_params.csv' ): 
        params = read_csv(self.dir_data+fname, sep =',',comment='#').astype(str).replace(' ', '', regex = True) 
        for val, field in zip([i.strip() for i in [samp,features,algorithm]],['samp','var','alg']):
            params = params[params[field]==val].copy().drop(field,axis = 1)
        params: list = params.astype(float).values.tolist()[0]
        if not params:       
            print('value not in csv')
            if features=='umap':#if umap is not in csv - use default umap values
                params = 0.1,10
        print(params)
        return params


    def umap(self,df :DataFrame,params:list = [])-> DataFrame:  
             
        if isfile(self.dir_data + f'umapData_{self.figname}.p') and not self.recalculate_umap: 
            umapData = self.pickle_load(f'umapData_{self.figname}', self.dir_data)
        else: #either we want to calculate umap again or the file does not exist
            params = self.read_csv( 'umap',self.j,self.name,) if not params else params #if no params are given by user as input - take from csv
            umapData = UMAP( n_neighbors=int(params[1]), min_dist=params[0],# verbose=True,
                            n_components=2, metric='euclidean', random_state=42,  densmap=False,).fit_transform(df[self.features].copy(),)
            self.pickle_dump(f'umapData_{self.figname}', umapData, self.dir_data)
        self.umapData = DataFrame(umapData, columns=['umap1', 'umap2']) 
        return self.umapData
    def calculate_dbscan(self,X,eps=0.1,min_samples=50):
        # DBSCAN - Density-Based Spatial Clustering of Applications with Noise. 
        # Finds core samples of high density and expands clusters from them. Good for data which contains clusters of similar density
        X = self.umapData.copy()
        X = StandardScaler().fit_transform(X)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        print('Estimated number of clusters: %d' % (len(set(labels)) - (1 if -1 in labels else 0)))
        print('Estimated number of noise points: %d' % list(labels).count(-1))
        # try:
        #     print("Silhouette Coefficient: %0.3f" % silhouette_score(X, labels))
        # except:
        #     print('Silhouette impossible; only 1 cluster recognized')
        
        return Series(labels), [X, core_samples_mask]
         
    def DBscan(self,cmap: list =[],params:list = [])-> Series:  
        if not isfile(self.dir_data + f'dbLabels_{self.figname}.p') or self.recalculate_db: 
            params  = self.read_csv( 'db',self.j,self.name,) if not params else params #if no params are given by user as input - take from csv
                
            labels, dbData  = self.calculate_dbscan(self.umapData, eps=params[0], min_samples=int(params[1]))
            colors = self.get_clustering_colors(labels,cmap)
            self.pickle_dump(f'dbLabels_{self.figname}', labels, self.dir_data); 
            self.pickle_dump(f'dbData_{self.figname}', dbData, self.dir_data); 
            self.pickle_dump(f'dbColors_{self.figname}', colors, self.dir_data)

        labels,self.dbData,self.colors = [self.pickle_load(f'{f}_{self.figname}', self.dir_data) for f in ['dbLabels','dbData','dbColors']]
        self.labels = Series(labels)
        return self.labels
    
    def get_clustering_colors(self,labels:Series,cmap: list =[])->list[tuple]:
        unique_labels = sorted(labels.unique())
        
        if len(unique_labels)-1>len(cmap): #too many clusters for the preselected colors map
            cmap = [plt.get_cmap('Set2')(each) for each in np.linspace(0, 1, len(unique_labels)-1)]# #cmap of Set2, twilight, PuOr and cividis.
        colors = [(0, 0, 0, 1)]+[cmap[each] for each in range(len(unique_labels)-1)]# noise (-1 label) is black color
        return colors    
    



        