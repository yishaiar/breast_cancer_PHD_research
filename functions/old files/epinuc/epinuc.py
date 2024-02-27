import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
import os
import sys
parent_dir = os.getcwd()
sys.path.append(parent_dir+'/functions/')
from plot_functions import *

def loadEpinucData(fname,sampleInd=np.arange(8)):
    dir  = "~/Dropbox/CyTOF_Breast/Kaplan-epinuc/"
    K=pd.read_csv(dir+fname)
    K  = K.copy().T[sampleInd].T
    K['by_sample'] = K['Subject'].astype(float)-130
    
    # K.set_index( K['Subject'].values)
    K = K.set_index('by_sample')

        # drop sample 3,6
    K = K.drop([3,6],axis =0)
    K.loc[4.1]=K.loc[4].copy()
    K.sort_index(inplace = True)

    # [['H3K4me1',	'H3K27me3',	'H3K9me3',	'H3K9ac',	'H3K4me3',	'H3K36me3']]\
    #     = K[['H3K4me1/Nuc',	'H3K27me3/Nuc',	'H3K9me3/Nuc',	'H3K9ac/Nuc',	'H3K4me3/Nuc',	'H3K36me3/Nuc']]
    
    cols = [ 'Subtype', 'tumor.type','TIMP1','Methylation','Subject']
    K.drop(columns = cols, inplace=True)



    K = K.rename(columns={'H3K4me1/Nuc':'H3K4me1','H3K27me3/Nuc':'H3K27me3',
                    'H3K9me3/Nuc':'H3K9me3','H3K9ac/Nuc':'H3K9ac',
                    'H3K4me3/Nuc':'H3K4me3','H3K36me3/Nuc':'H3K36me3'}) 
    return K
def splitDf(df,f):
    newDf = pd.DataFrame(index=df.index)

    newDf[f] = df [f].copy().astype(float)
    return newDf

def createRawMat(k,kEpinuc,features =[],):
    k = splitDf(k, features)
    kEpinuc = splitDf(kEpinuc, features)
    kEpinuc.index= kEpinuc.index.astype(str)+'_epinuc'
    k.index = k.index.astype(str)+'_cytof'
    rawMat = kEpinuc.append(k)   
    return rawMat 

def epinucFeaturesInCytof(k):
    newDf = pd.DataFrame(index=k.index)
    try:
        k = k[k['by_sample']!=11].reset_index(drop=True)
        newDf['by_sample'] = k['by_sample']
    except:
        pass
    # global
    f =['H3K4me1',	'H3K27me3',	'H3K9me3',	'H3K9ac',	'H3K4me3',	'H3K36me3']
    newDf[f] = k[f]
    # relative
    newDf['H3K9ac/H3K4me1'] = k['H3K9ac']/k['H3K4me1']
    newDf['H3K4me3/H3K27me3'] = k['H3K4me3']/k['H3K27me3']
    newDf['H3K36me3/H3K9me3'] = k['H3K36me3']/k['H3K9me3']
    # additive
    newDf['H3K4me1+H3K9ac'] = k['H3K4me1']+k['H3K9ac']
    newDf['H3K27me3+H3K4me3'] = k['H3K27me3']+k['H3K4me3']
    newDf['H3K9me3+H3K36me3'] = k['H3K9me3']+k['H3K36me3']
    

    return newDf

if __name__ == "__main__":
    # import os
    from usefull_functions import *
    # parent_dir = os.getcwd()
    dir_data = parent_dir+'/Data/'
    
    
    kEpinuc =  loadEpinucData("EPINUC_BCK.csv")
    # # tuple of raw data; 1- Xi,2- Xi/Xj,3- Xi+Xj
    # EpinucData =  CreateRawMats(kEpinuc)


    k_ =pickle_load('kb123_dict',dir_data )['k']
    
    # k['by_sample'] = k['by_sample'] .astype(int)
    k= epinucFeaturesInCytof(k_.copy())
    # normalize k..

    # average over epinuc features after normalizing them
    # caculate anothr dataset which is the sum of the average rather than the average of the sum
    kBeforeAverage = k.groupby(by='by_sample').mean()
    kAfterAverage = epinucFeaturesInCytof(kBeforeAverage[kBeforeAverage.columns[:6]].copy())
    # kAfterAverage =  CreateRawMats(kAfterAverage)
    # kBeforeAverage =  CreateRawMats(kBeforeAverage)
    # calculate the corr mats between mats in the tuples (1--1,2--2,3--3)
    # output for each calculation is the raw merged mat,corrMAT and the transposed cormat
    createCorrMat(createRawMat(kBeforeAverage,kEpinuc,
                  ['H3K4me1','H3K27me3','H3K9me3',	
                'H3K9ac','H3K4me3','H3K36me3']))
    
    createCorrMat(kAfterAverage,kEpinuc,)
    # print(Kepinuc)
    # print(Kepinuc.corr(method ='pearson'))
    print()

    print(1)

# # group_name,group = groups[0][-1],groups[1][-1]
# # HeatMap(k,group,settings,clustFeature='by_sample',
# #         title = 'T'+j+' Cell Iden Based: '+group_name+'by_sample',
# #         figname = name+'1_HeatMap_'+group_name+'by_sample')
# Mat=   k.groupby(by='by_sample').mean()[Kepinuc.columns]
# Mat.index = Mat.index.astype(int).astype(str)+'_cytof'
# Kepinuc = Kepinuc.append(Mat)
# # print(Kepinuc)
# # print(Kepinuc.T.corr())


# g=sns.clustermap(Kepinuc.T.corr(),cmap=plt.cm.seismic,
#                 #  vmin=amin,vmax=amax,
#                 # figsize=(10,20), annot_kws={"size":8}, center=0,
#                 figsize=(6, 10), annot_kws={"size":8}, center=0,
#                 annot=True, linewidths=1,linecolor='k',)