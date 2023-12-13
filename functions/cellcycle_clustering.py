
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
def split_into_unique_samples(samples,f='samp'):#split into list of uique i.e 4,4,1,7,8,8.1 etc
    samples_list = []
    for i in samples[f].unique():   

        samples_list.append(samples[samples[f]==i].copy())
    return samples_list
def get_class(cls,num = True):
    if 'basal' in cls:
        return 0 if num else 'basal'
    elif 'luminal' in cls:
        return 1 if num else 'luminal'
    elif 'Cycling' in cls:
        return 2 if num else 'Cycling'
    elif 'unknown' in cls:
        return 3 if num else 'unknown'
    elif '-1' in cls or -1 in cls:
        return -1 if num else '-1'
        

def class_dict(add,num = True):
    dict = {}

    df = pd.read_csv(add).dropna(axis=1, how='all').dropna(axis=0, how='all')

    # find total clusters of each sample and drop row from df 
    s = df['samp'].unique()
    # t = [int(df[np.asarray(df['samp']==samp) * np.asarray(df['class']=='total')]['clusters']) for samp in s]
    t = [15 for samp in s]
    df = df[df['class']!='total']


    for samp,total_clusters_ in zip(s,t):#iterate on all samples
        # if samp !=5:
        #     continue
        total_clusters = np.arange(total_clusters_+1) # all clusters in sample  
        samp_clusters = []
        for i in df[df['samp']==samp].index:
            if 'rest' not in df.loc[i]['clusters'] : 
                for cluster in np.asarray(df.loc[i]['clusters'].split(';')).astype(int):    
                    dict[(samp,cluster)] = df.loc[i]['class']
                    samp_clusters.append(cluster)
            else:#save index of row with rest info
                ind = i
        for cluster in [t for t in total_clusters if t not in samp_clusters]:#rest clusters
            dict[(samp,cluster)] = df.loc[ind]['class']
        # add -1
        dict[(samp,-1)] = '-1'

    for key in dict.keys():
        dict[key] = get_class(dict[key],num = num)
    return dict
if __name__ == "__main__": 
    # from cellcycle_clustering import *   
    dict = class_dict(add = '/home/yishai/Dropbox/CyTOF_Breast/Kaplan_5th/_clusters.csv')
