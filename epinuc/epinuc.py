import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
from plot_functions import *

def loadEpinucData(fname,sampleMax=5):
    dir  = "~/Dropbox/CyTOF_Breast/Kaplan-epinuc/"
    K=pd.read_csv(dir+fname)
    K = K.drop(np.arange(sampleMax,len(K)),axis =0)
    K['Subject'] = K['Subject'].astype(int)-130
    
    # K.set_index( K['Subject'].values)
    K = K.set_index('Subject')
    K.index.name='by_sample'

    # [['H3K4me1',	'H3K27me3',	'H3K9me3',	'H3K9ac',	'H3K4me3',	'H3K36me3']]\
    #     = K[['H3K4me1/Nuc',	'H3K27me3/Nuc',	'H3K9me3/Nuc',	'H3K9ac/Nuc',	'H3K4me3/Nuc',	'H3K36me3/Nuc']]
    
    cols = [ 'Subtype', 'tumor.type','TIMP1','Methylation']
    K.drop(columns = cols, inplace=True)

    # drop sample 3
    K = K.drop(3,axis =0)

    K = K.rename(columns={'H3K4me1/Nuc':'H3K4me1','H3K27me3/Nuc':'H3K27me3',
                    'H3K9me3/Nuc':'H3K9me3','H3K9ac/Nuc':'H3K9ac',
                    'H3K4me3/Nuc':'H3K4me3','H3K36me3/Nuc':'H3K36me3'}) 
    return K
    
def CreateRawMats(K):    
    # Global levels of the modification - normlized to number of nucleosomes. 
    # May relate to the mean levels of modification identified in tumor/subcluster
    LevelsGlobal = pd.DataFrame(index=K.index)
    LevelsGlobal    [['H3K4me1',	'H3K27me3',	'H3K9me3',	'H3K9ac',	'H3K4me3',	'H3K36me3']]\
            = K     [['H3K4me1',	'H3K27me3',	'H3K9me3',	'H3K9ac',	'H3K4me3',	'H3K36me3']]
    # ratio
    LevelsRelative = pd.DataFrame(index=K.index)
    LevelsRelative [['H3K9ac/H3K4me1', 'H3K4me3/H3K27me3', 'H3K36me3/H3K9me3']]\
        = K [['H3K9ac/H3K4me1', 'H3K4me3/H3K27me3', 'H3K36me3/H3K9me3']]
    
    LevelsAdditive = pd.DataFrame(index=K.index)
    LevelsAdditive [['H3K4me1+H3K9ac', 'H3K27me3+H3K4me3', 'H3K9me3+H3K36me3']]\
        = K [['H3K4me1+H3K9ac', 'H3K27me3+H3K4me3', 'H3K9me3+H3K36me3']]


    # print (globalLeveles)
    return LevelsGlobal.astype(float),LevelsRelative.astype(float), LevelsAdditive.astype(float)

def createCorrMat(k,kEpinuc,settings,title='',figname = '',method ='spearman'):
    
    
    kEpinuc.index= kEpinuc.index.astype(str)+'_epinuc'
    k.index = k.index.astype(str)+'_cytof'
    rawMat = kEpinuc.append(k)
    corrMat = rawMat.corr(method =method)
    corrMatSamples = rawMat.T.corr(method =method)

    # plotHeatMap(rawMat,'raw;'+title,settings,'raw_'+figname)
    rawMat.to_csv(settings[0]+'raw_'+figname+'.csv')
    plotHeatMap(corrMat,'corrFeatures;'+title,settings,'corrFeatures_'+figname)
    plotHeatMap(corrMatSamples,'corrSamples;'+title,settings,'corrSamples_'+figname)
        # plot matrices..

    # return kEpinuc,corrMat,corrMatSamples
def epinucFeaturesInCytof(k):
    
    # relative
    k['H3K9ac/H3K4me1'] = k['H3K9ac']/k['H3K4me1']
    k['H3K4me3/H3K27me3'] = k['H3K4me3']/k['H3K27me3']
    k['H3K36me3/H3K9me3'] = k['H3K36me3']/k['H3K9me3']
    # additive
    k['H3K4me1+H3K9ac'] = k['H3K4me1']+k['H3K9ac']
    k['H3K27me3+H3K4me3'] = k['H3K27me3']+k['H3K4me3']
    k['H3K9me3+H3K36me3'] = k['H3K9me3']+k['H3K36me3']
    
    features = [
                'H3K4me1',	'H3K27me3',	'H3K9me3',	'H3K9ac',	'H3K4me3',	'H3K36me3',
                'H3K9ac/H3K4me1', 'H3K4me3/H3K27me3','H3K36me3/H3K9me3',
                'H3K4me1+H3K9ac','H3K27me3+H3K4me3','H3K9me3+H3K36me3'
                ]
    return k,features
def averageOverFeatures(k,features):
    
    
    return kBeforeAverage,kAfterAverage
if __name__ == "__main__":
    import os
    from usefull_functions import *
    parent_dir = os.getcwd()
    dir_data = parent_dir+'/Data/'
    k =pickle_load('k1245_dict',dir_data )['k']
    k['by_sample'] = k['by_sample'] .astype(int)
    
    kEpinuc =  loadEpinucData("EPINUC_BCK.csv",sampleMax=5)
    # tuple of raw data; 1- Xi,2- Xi/Xj,3- Xi+Xj
    EpinucData =  CreateRawMats(kEpinuc)


    k,neededFeatures = epinucFeaturesInCytof(k)
    # normalize k..

    # average over epinuc features after normalizing them
    # caculate anothr dataset which is the sum of the average rather than the average of the sum
    kBeforeAverage = k.groupby(by='by_sample').mean()[neededFeatures]
    kAfterAverage,_ = epinucFeaturesInCytof(kBeforeAverage[neededFeatures[:6]].copy())
    kAfterAverage =  CreateRawMats(kAfterAverage)
    kBeforeAverage =  CreateRawMats(kBeforeAverage)
    # calculate the corr mats between mats in the tuples (1--1,2--2,3--3)
    # output for each calculation is the raw merged mat,corrMAT and the transposed cormat
    createCorrMat(kBeforeAverage,EpinucData)
    createCorrMat(kAfterAverage,EpinucData)
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