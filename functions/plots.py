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

from numpy import linspace

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D


class Plots(Parent):
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)#import keys and values from dictionary to class
        # self.get_attribute()#print all attributes
        # self.calc_db = False
        self.folderExists(self.dir_plots)#verify that plots folder exists
        
    def get_clustering_colors(self,labels:Series,cmap: list =[])->list[tuple]:
        unique_labels = sorted(labels.unique())
        
        if len(unique_labels)-1>len(cmap): #too many clusters for the preselected colors map
            cmap = [plt.get_cmap('Set2')(each) for each in linspace(0, 1, len(unique_labels)-1)]# #cmap of Set2, twilight, PuOr and cividis.
        colors = [(0, 0, 0, 1)]+[cmap[each] for each in range(len(unique_labels)-1)]# noise (-1 label) is black color
        return colors    
    def umap(self, umapData:DataFrame,intensity:Series=None,ind:list[int] = [],title:str = '',figname :str= '',backgroundColor :str= 'gainsboro') -> None:#limits = [None,None,None,None]
        '''
        plot umap with intensity according to specific feature values in each point in map
        umapData - umap coordinates of each point in sample
        intensity - feature values of each point in sample (same order as umapData)
        
        '''

        fig,axs = plt.subplots(1,figsize = (6, 5))

        axs.set_ylabel('umap2')
        axs.set_xlabel('umap1')
        axs.set_title(title)  
        
        # plot settings
        axs.set_facecolor(backgroundColor)
        # if not intensity:
        #     intensity = umapData.copy().mean(axis=1)
        vmax=intensity.quantile(0.99);vmin=intensity.quantile(0.01)
        
        # colorbar settings
        norm = Normalize(vmax=vmax,vmin=vmin)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=plt.cm.seismic),ax  = axs)
        
        
        ind = list(intensity.index) if not ind else ind# if ind is empty take all indexes
        axs.scatter(umapData['umap1'].loc[ind],umapData['umap2'].loc[ind],c=intensity.loc[ind],
                    s=2, cmap=plt.cm.seismic, vmax=vmax,vmin=vmin) 
   
        self.figSettings(fig,figname)
        
        # if limits[0] is not  None:
        #     axs.set_xlim(limits[0], limits[2])  
        #     axs.set_ylim(limits[1], limits[3])
        # else:
        #     axs.set_yticklabels([])
        #     axs.set_xticklabels([])      
        
    def labeled_umap(self, umapData:DataFrame,labels:Series,ind:list[int] = [],title:str = '',figname :str= '',backgroundColor :str= 'gainsboro',colors :list[str]=[],UNIQ = None) -> None:#limits = [None,None,None,None]
        '''
        plot umap according to label values in each point in map
        umapData - umap coordinates of each point in sample
        labels - label values of each point in sample (same order as umapData)
        
        points without label are drawn in white-gray
        labels should be a series with the same index as umapData but smaller length (only the points with labels)
        
        '''
        
        if not colors or len(colors)<len(labels.unique()):#colors not given by user - #cmap of Set2, twilight, PuOr and cividis.
            colors = [plt.get_cmap('Set2')(each) for each in linspace(0, 1, len(labels.unique())+1)]#[:-1]]# avoid color 1 - cmap('set2)(1) gray same as background
        if -1 in labels.unique() and len(colors)>=len(labels.unique()):# there is noise
            colors = [(0, 0, 0, 1)]+ colors[:-1]# noise (-1 label) is black color (since we are using sorted in the coloring afterward: noise -1 is first)

        fig,axs = plt.subplots(1,figsize = (10, 5))

        axs.set_ylabel('umap2')
        axs.set_xlabel('umap1')
        axs.set_title(title)  
        
        

        # colors = cm.rainbow((labels+1)/np.max(labels+1))
        
        ind = umapData.loc[~umapData.index.isin(labels.index)].index#drop points without label
        axs.scatter(umapData['umap1'].loc[ind],umapData['umap2'].loc[ind],c=backgroundColor, alpha=0.2,s=2)#c = backgroundColor
        UNIQ = sorted(labels.unique()) if UNIQ is None else UNIQ
        for i,uniq in enumerate(UNIQ):#uniq values(clusters in sample)
            print(i)
            ind = labels[labels==uniq].index
            axs.scatter(umapData['umap1'].loc[ind],umapData['umap2'].loc[ind], s=2,label = uniq,color = colors[i])#,c = cc[cluster],
            # print(uniq)


        axs.legend(fontsize=15, title_fontsize='40',markerscale = 3.5,ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True,) #
   
        self.figSettings(fig,figname)
        
        # if limits[0] is not  None:
        #     axs.set_xlim(limits[0], limits[2])  
        #     axs.set_ylim(limits[1], limits[3])
        # else:
        #     axs.set_yticklabels([])
        #     axs.set_xticklabels([]) 
    # def labeled_umap_by_sample(self,umapData:DataFrame, sample_labels:Series,title:str = '',figname :str= '',backgroundColor :str= 'gainsboro',colors :list[str]=[]):
                              
    
    #     umapData = umapData.loc[sample_labels.index].reset_index(drop=True)
    #     sample_labels = sample_labels.reset_index(drop=True)

            
            
    #     if not colors:#colors not given by user
    #         colors = [plt.get_cmap('Set2')(each) for each in np.linspace(0, 1, len(sample_labels.unique()))]# #cmap of Set2, twilight, PuOr and cividis.
    #     if -1 in sample_labels.unique() and len(colors)>=len(sample_labels.unique()):# there is noise
    #         colors = [(0, 0, 0, 1)]+ colors[:-1]# noise (-1 label) is black color

    #     fig,axs = plt.subplots(1,figsize = (6, 5))

    #     axs.set_ylabel('umap2')
    #     axs.set_xlabel('umap1')
    #     axs.set_title(title)  

    #     # plt.figure(figsize=(10, 10))
    #     # axs.scatter(umapData['umap1'],umapData['umap2'],c = backgroundColor, alpha=0.2,s=2)
        
    #     ind = umapData.loc[~umapData.index.isin(sample_labels.index)].index#points without label
    #     axs.scatter(umapData['umap1'].loc[ind],umapData['umap2'].loc[ind],c=backgroundColor, alpha=0.2,s=2)#c = backgroundColor#c='lightgray'

    #     for i,uniq in enumerate(sorted(sample_labels.unique())):#uniq values(clusters in sample)
    #         cluster_ind = sample_labels[sample_labels==uniq].index
    #         axs.scatter(umapData.loc[cluster_ind]['umap1'],umapData.loc[cluster_ind]['umap2'],color = colors[i],s=2,label = uniq)#alpha=0.5
        
    #     axs.set_title(title)

    #     axs.legend(fontsize=15, title_fontsize='40',markerscale = 3.5,ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True,) #   
    #     self.figSettings(fig,figname)    
        
    def umap_by_feature(self,umapData:DataFrame,df:DataFrame, features:list[str], ind:list[int] = [] ,title:str = '',figname:str = '',backgroundColor:str  = 'gainsboro'): 
        '''
        plot umap with intensity according to specific feature values in each point in map
        data - dataframe with feature values of each point in sample (same order as umapData)
        umapData - umap coordinates of each point in sample
        intensity - feature values of each point in sample (same order as umapData)
        '''
        for feature in features:
            intensity = df[feature]
            self.umap(umapData.copy(),intensity,ind = ind,title = f'{title}- {feature}',figname = figname + feature,backgroundColor = backgroundColor)


 
    def DBscan(self,labels:Series,dbData,colors,title:str = '',figname :str= '') -> None:
        '''
        X - StandardScaler scaled umap coordinates of each point in sample
        core_samples_mask - boolean array of core points
        
        '''
        # c = colors.copy()
        # colors = c.copy()
        if -1 not in labels.unique() and len(colors)>len(labels.unique()):
            colors = colors[1:len(labels.unique())+1 ]#drop the noise color
        X,core_samples_mask = dbData
        # drop points without label (such when noise is dropped)
        X = DataFrame(X, columns=['db1', 'db2']).loc[labels.index] 
        core_samples_mask = Series(core_samples_mask).loc[labels.index] 

        fig,axs = plt.subplots(1,figsize = (6, 5))
        
        axs.set_ylabel('db2')
        axs.set_xlabel('db1')
        axs.set_title(title)  
        
        for label, color in zip(sorted(labels.unique()), colors):
            if label ==7:
                print(1)
            class_member_mask = labels == label
            
            # cluster edges with smaller marker size for finess
            xy : DataFrame = X[class_member_mask & ~core_samples_mask]
            axs.plot(xy['db1'], xy['db2'], 
                    'o', markerfacecolor=tuple(color),markeredgecolor='k', markersize=6,
                    )
            
            xy : DataFrame = X[class_member_mask & core_samples_mask]
            axs.plot(xy['db1'], xy['db2'], 
                     'o', markerfacecolor=tuple(color),markeredgecolor='k', markersize=14,
                    label = label,# add label to legend  
                    )
            

            
        def update(handle, orig):
            handle.update_from(orig)
            handle.set_alpha(1)
        axs.legend(fontsize=15, title_fontsize='40',
            loc='upper center', 
            bbox_to_anchor=(0.5, -0.05),
            # fancybox=True, 
            # shadow=False, 
            ncol=5,

            handler_map={PathCollection : HandlerPathCollection(update_func= update),#update legend alpha
            plt.Line2D : HandlerLine2D(update_func = update)}
            )


        self.figSettings(fig,figname)
        





