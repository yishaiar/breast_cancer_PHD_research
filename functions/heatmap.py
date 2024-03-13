from parent_class import * 

import numpy as np 
import pandas as pd 
from seaborn import clustermap
import matplotlib.pyplot as plt


from pandas import DataFrame,Series
# from typing import List
import numpy as np
from numpy import ndarray
from matplotlib.pyplot import figure

class Heatmap(Parent):
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)#import keys and values from dictionary to class
        # self.get_attribute()#print all attributes
        # self.calc_db = False
        self.folderExists(self.dir_plots)#verify that plots folder exists
        
        # CellCycle feature is only in the '.2' data
        self.features_groups = ['CellIden', 'EpiCols', 'CellIden+EpiCols'] if '.2' not in self.j else ['CellIden', 'EpiCols', 'CellIden+EpiCols'] + ['CellCycle']
        
    def plot(self, df: DataFrame,labels:Series,features:list[str],ind:list[int] = [],title:str = '',figname:str = '' ,amin = None,amax = None,rot_tentogram = False) -> None:
        '''
        calculates the mean of the features for each label and saves it as a csv file
        load the csv file and plot it as a heatmap
        mat - the heatmap matrix
        '''
        self.to_csv( df,labels,features,ind,figname )
        mat = pd.read_csv(self.dir_plots+figname+'.csv',index_col=0)
        self.to_heatmap(mat,title,figname,amin,amax,rot_tentogram)
         
    def to_heatmap(self,mat:DataFrame,title:str = '',figname:str = '' ,amin = None,amax = None,rot_tentogram = False) -> None:
        
        
        
        figsize = (max(10, int(np.ceil(len(mat.index)*0.7))), max(10, int(np.ceil(len(mat.columns)*0.7))))
        # amin,amax - defines the  colormap dynamic range
        amin=mat.min().min() if amin is None else amin
        amax=mat.max().max() if amax is None else amax
        

        if rot_tentogram:
            row_cluster,col_cluster=True,False,
        else:    
            row_cluster,col_cluster=False,True,

        # fig,axs = plt.subplots(1,figsize=(10, 10))
        # g is a seaborn clustermap (not a figure)
        g = clustermap(mat,cmap=plt.cm.seismic,vmin=amin,vmax=amax,
                        row_cluster=row_cluster,col_cluster=col_cluster,
                        figsize=figsize, annot_kws={"size":8}, center=0,
                        annot=True, linewidths=1,linecolor='k')
        g.ax_col_dendrogram.set_title(title) 
        # plt.show()
        self.figSettings(g.fig,figname,)
        
        
        
    def to_csv(self, df: DataFrame,labels:Series,features:list[str],ind:list[int] = [],figname:str = '' ) -> None:
        '''
        calculates the mean of the features for each label and saves it as a csv file
        used for plotting heatmaps of the data
        
        drop anchor data columns (with '.1' in the name) if they exist and are not all the columns of the dataframe
        
        labels - the labels of the data (same size a the data rows index; label for each row)
        '''
        
        ind = list(df.index) if not ind else ind# if ind is empty take all indexes
        mat = df.loc[ind].groupby(by=labels.loc[ind]).mean(numeric_only=True)[features].T
        
        drop_cols = [i for i in mat.columns if '.1'  in str(i)]
        drop_cols = drop_cols if len(drop_cols) < len(mat.columns) else []
        
        mat.drop(drop_cols,axis = 1,inplace=True)
        mat = mat.loc[sorted(mat.index)]
        mat = mat[sorted(mat.columns)]
        mat.to_csv(self.dir_plots+figname+'.csv')
        
        # plotHeatMap(Mat,title,settings,figname,amin = amin,amax = amax,rot_tentogram = rot_tentogram)
    
# heatmaps = Heatmaps(**config)

