
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize

# from matplotlib.collections import PathCollection
# from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D

# from os.path import isfile


# from pandas import DataFrame,Series
# # from typing import List
# import numpy as np
# from numpy import ndarray
# from matplotlib.pyplot import figure

# from pandas import read_csv
# from umap import UMAP

# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import  DBSCAN

# from parent_class import * 







# class Umap_dbscan(Parent):
    
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)#import keys and values from dictionary to class
#         # self.get_attribute()#print all attributes
#         # self.calc_db = False
#         self.folderExists(self.dir_plots)#verify that plots folder exists
#     def read_csv(self,algorithm,samp,features,fname = '_params.csv' ): 
#         params = read_csv(self.dir_data+fname, sep =',',comment='#').astype(str).replace(' ', '', regex = True) 
#         for val, field in zip([i.strip() for i in [samp,features,algorithm]],['samp','var','alg']):
#             params = params[params[field]==val].copy().drop(field,axis = 1)
#         params: list = params.astype(float).values.tolist()[0]
#         if not params:       
#             print('value not in csv')
#             if features=='umap':#if umap is not in csv - use default umap values
#                 params = 0.1,10
#         print(params)
#         return params


#     def get_umapData(self,df :DataFrame,params:list = [])-> DataFrame:  
             
#         if isfile(self.dir_data + f'umapData_{self.figname}.p') and not self.calc_umap  : 
#             umapData = self.pickle_load(f'umapData_{self.figname}', self.dir_data)
#         else: #either we want to calculate umap again or the file does not exist
#             if not params: #if no params are given by user - take from csv
#                 #umap params are [min_dist, n_neighbors]
#                 params = self.read_csv( 'umap',self.j,self.name,)

#             umapData = UMAP( n_neighbors=int(params[1]), min_dist=params[0],# verbose=True,
#                             n_components=2, metric='euclidean', random_state=42,  densmap=False,).fit_transform(df[self.features].copy(),)
#             self.pickle_dump(f'umapData_{self.figname}', umapData, self.dir_data)
#         self.umapData = DataFrame(umapData, columns=['umap1', 'umap2']) 
#         return self.umapData
#     def calculate_dbscan(self,X,eps=0.1,min_samples=50):
#         # DBSCAN - Density-Based Spatial Clustering of Applications with Noise. 
#         # Finds core samples of high density and expands clusters from them. Good for data which contains clusters of similar density
#         X = self.umapData.copy()
#         X = StandardScaler().fit_transform(X)
#         db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
#         core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#         core_samples_mask[db.core_sample_indices_] = True
#         labels = db.labels_

#         # Number of clusters in labels, ignoring noise if present.
#         print('Estimated number of clusters: %d' % (len(set(labels)) - (1 if -1 in labels else 0)))
#         print('Estimated number of noise points: %d' % list(labels).count(-1))
#         # try:
#         #     print("Silhouette Coefficient: %0.3f" % silhouette_score(X, labels))
#         # except:
#         #     print('Silhouette impossible; only 1 cluster recognized')
        
#         return Series(labels), [X, core_samples_mask]
         
#     def get_dbData(self,cmap: list =[],params:list = [])-> Series:  
#         if not isfile(self.dir_data + f'dbLabels_{self.figname}.p') or self.calc_db: 
#             if not params: #if no params are given by user - take from csv
#                 params = self.read_csv( 'db',self.j,self.name,)
                
#             labels, dbData  = self.calculate_dbscan(self.umapData, eps=params[0], min_samples=int(params[1]))
#             colors = self.cluster_colors(labels,cmap)
#             self.pickle_dump(f'dbLabels_{self.figname}', labels, self.dir_data); 
#             self.pickle_dump(f'dbData_{self.figname}', dbData, self.dir_data); 
#             self.pickle_dump(f'dbColors_{self.figname}', colors, self.dir_data)

#         labels,self.dbData,self.colors = [self.pickle_load(f'{i}_{self.figname}', self.dir_data) for i in ['dbLabels','dbData','dbColors']]
#         self.labels = Series(labels)
#         return self.labels
#     def cluster_colors(self,labels:Series,cmap: list =[])->list[tuple]:
#         unique_labels = sorted(labels.unique())
        
#         if len(unique_labels)-1>len(cmap): #too many clusters for the preselected colors map
#             cmap = [plt.get_cmap('Set2')(each) for each in np.linspace(0, 1, len(unique_labels)-1)]# #cmap of Set2, twilight, PuOr and cividis.
#         colors = [(0, 0, 0, 1)]+[cmap[each] for each in range(len(unique_labels)-1)]# noise (-1 label) is black color
#         return colors    
#     def plot_umap(self, intensity:Series,ind:list[int] = [],title:str = '',figname :str= '',backgroundColor :str= 'gainsboro') -> None:#limits = [None,None,None,None]
#         '''
#         plot umap with intensity according to specific feature values in each point in map
#         umapData - umap coordinates of each point in sample
#         intensity - feature values of each point in sample (same order as umapData)
        
#         '''

#         fig,axs = plt.subplots(1,figsize = (6, 5))

#         axs.set_ylabel('umap2')
#         axs.set_xlabel('umap1')
#         axs.set_title(title)  
        
#         # plot settings
#         axs.set_facecolor(backgroundColor)
#         vmax=intensity.quantile(0.99);vmin=intensity.quantile(0.01)
        
#         # colorbar settings
#         norm = Normalize(vmax=vmax,vmin=vmin)
#         fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=plt.cm.seismic),ax  = axs)
        
        
#         ind = list(intensity.index) if not ind else ind# if ind is empty take all indexes
#         axs.scatter(self.umapData['umap1'].loc[ind],self.umapData['umap2'].loc[ind],c=intensity.loc[ind],
#                     s=2, cmap=plt.cm.seismic, vmax=vmax,vmin=vmin) 
   
#         self.figSettings(fig,figname)
        
#         # if limits[0] is not  None:
#         #     axs.set_xlim(limits[0], limits[2])  
#         #     axs.set_ylim(limits[1], limits[3])
#         # else:
#         #     axs.set_yticklabels([])
#         #     axs.set_xticklabels([])      
        
        
        
#     def plot_umap_by_feature(self,df:DataFrame, features:list[str], ind:list[int] = [] ,title:str = '',figname:str = '',backgroundColor:str  = 'gainsboro'): 
#         '''
#         plot umap with intensity according to specific feature values in each point in map
#         data - dataframe with feature values of each point in sample (same order as umapData)
#         umapData - umap coordinates of each point in sample
#         intensity - feature values of each point in sample (same order as umapData)
#         '''
#         for feature in features:
#             intensity = df[feature]
#             self.plot_umap(intensity,ind = ind,title = f'{title}- {feature}',figname = figname + feature,backgroundColor = backgroundColor)


 
#     def plot_db(self,title:str = '',figname :str= '') -> None:
#         '''
#         X - StandardScaler scaled umap coordinates of each point in sample
#         core_samples_mask - boolean array of core points
        
#         '''
#         X,core_samples_mask = self.dbData
#         X = DataFrame(X, columns=['db1', 'db2']) 
#         core_samples_mask = Series(core_samples_mask)

#         fig,axs = plt.subplots(1,figsize = (6, 5))
        
#         axs.set_ylabel('db2')
#         axs.set_xlabel('db1')
#         axs.set_title(title)  
        
#         for label, color in zip(sorted(self.labels.unique()), self.colors):
#             class_member_mask = self.labels == label
            
#             # cluster edges with smaller marker size for finess
#             xy : DataFrame = X[class_member_mask & ~core_samples_mask]
#             axs.plot(xy['db1'], xy['db2'], 
#                     'o', markerfacecolor=tuple(color),markeredgecolor='k', markersize=6,
#                     )
            
#             xy : DataFrame = X[class_member_mask & core_samples_mask]
#             axs.plot(xy['db1'], xy['db2'], 
#                      'o', markerfacecolor=tuple(color),markeredgecolor='k', markersize=14,
#                     label = label,# add label to legend  
#                     )
            

            
#         def update(handle, orig):
#             handle.update_from(orig)
#             handle.set_alpha(1)
#         axs.legend(fontsize=15, title_fontsize='40',
#             loc='upper center', 
#             bbox_to_anchor=(0.5, -0.05),
#             # fancybox=True, 
#             # shadow=False, 
#             ncol=5,

#             handler_map={PathCollection : HandlerPathCollection(update_func= update),#update legend alpha
#             plt.Line2D : HandlerLine2D(update_func = update)}
#             )


#         self.figSettings(fig,figname)