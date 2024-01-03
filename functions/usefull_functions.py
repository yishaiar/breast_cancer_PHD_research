import os
import pickle 
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    
def deleteVars(to_delete=[]):
  for _var in to_delete:
      if _var in locals() or _var in globals():
          exec(f'del {_var}')
          
          
def removeFeatures(dict,remove_features =['']):
  
  for key in dict.keys():
    if dict[key] is not None:
        list_ = dict[key].copy()

        for feature in remove_features:
            try:
                list_.remove(feature)
            except:
                pass
        dict[key] = list_
  return dict

def createAdjustedDataset(k,j,dbscanData,labels,core_samples_mask,settings,figname = '',
                          clustersToRemove=[],limits = (None,None,None,None),drawFunc = None,
                          contained_samples = False
                          ):
    # filter by clusters to remove and filter by dbscanData limits
    xMin, xMax,yMin, yMax = limits

    # filter by clusters to remove
    idx1 = ~np.zeros_like(labels).astype(bool)
    for cluster in clustersToRemove:
        idx1 *= ~(labels == cluster)

    # filter by dbscanData
    xMin = xMin if xMin is not None else np.min(dbscanData[:,0])
    xMax = xMax if xMax is not None else np.max(dbscanData[:,0])
    yMin = yMin if yMin is not None else np.min(dbscanData[:,1])
    yMax = yMax if yMax is not None else np.max(dbscanData[:,1])
    idx2 = (dbscanData[:,0]<=xMax)*(dbscanData[:,0]>=xMin)*(dbscanData[:,1]<=yMax)*(dbscanData[:,1]>=yMin)

    # indexes to keep
    idx3 = idx1*idx2
    idx = np.asarray([i for i, j in enumerate(idx3) if j])
    idx4 = np.asarray([i for i, j in enumerate(idx3) if not j])
    print (f'k{j}; samples = {len(idx)}; saved to file')



    _ = drawFunc(dbscanData[idx], labels[idx], core_samples_mask[idx], settings, figname='1_'+figname+'dbscan_adjusted')
    _ = drawFunc(dbscanData[idx4], labels[idx4], core_samples_mask[idx4], settings, figname='1_'+figname+'dbscan_adjusted_stroma')

    
    indArr = {}
    # index according to original data
    newK = k.copy().reset_index(drop = True).loc[idx]
    ind = np.asarray(newK['ind'].copy()) 
    kInd = f'{j}a'
    indArr[kInd]=ind

    
    if contained_samples:# # create adjusted for all contained samples
        for kInd in newK['samp'].unique():
            newK_ = newK[newK['samp']==kInd].copy()
            ind = np.asarray(newK_['ind'].copy())
            kInd = int(kInd) if kInd%1==0.0 else float(kInd);kInd = f'{kInd}a'
            indArr[kInd]=ind
    return indArr,idx,idx4

    







def pickle_dump(file_name, dict,dir_data):
  with open(dir_data+file_name+'.p', "wb") as f:
    pickle.dump(dict, f, protocol=pickle.HIGHEST_PROTOCOL)


    
def pickle_load(file_name,dir_data):    
  with open(dir_data  + file_name + '.p', 'rb') as f:
      dict_ = pickle.load(f)
      print(file_name,'; loaded from file')
      return dict_
  
  
def folderExists(path,prompt = True):
  if not os.path.exists(path):
   # Create a new directory because it does not exist
   os.makedirs(path)
   if prompt:
    print("The new directory is created!")
   
   
def test_fetures(name):
    NamesAll = name['NamesAll'].copy()
    CellIden = name['CellIden'].copy()
    EpiCols = name['EpiCols'].copy()
    Core = name['Core'].copy()
 
    new_NamesAll = CellIden+EpiCols + Core
    test1 = all(item in  NamesAll  for item in new_NamesAll)
    test3 = all(item in  new_NamesAll  for item in NamesAll)
    # print (test1)
    # if not test1:
    #     for i in new_NamesAll:
    #         NamesAll.remove(i)
        
    # print(NamesAll)

    test2 = not (any(item in CellIden for item in EpiCols))
    return test1*test2*test3


    
from os import listdir
from os.path import isfile, join
# from fpdf import FPDF




# cairosvg.svg2png(url=im_name, write_to=im_name+'.png')

# def imList2pdf(dir_plots,j,groups):
        
#     print(dir_plots)
    
#     totalImagelist = [ join(dir_plots, f) for f in listdir(dir_plots) if isfile(join(dir_plots, f)) and f.endswith ('png')]
#     for group in groups:
#         name = j+'_'+group+'_'
#         print(name)
#         imagelist = [ f for f in totalImagelist if name in f]
#         imagelist.sort()
        

#         pdf = FPDF()
#         pdf.add_page()

#         for im_name in tqdm(imagelist):
            
#             pdf.image(im_name,x = 0,y = 50,w = 210,h = 210)
#             # counter +=1
#             pdf.add_page()           
#         pdf.output(dir_plots+name+'.pdf', "F")  
# 
#  
    # counter = 0   
    # for im_name in tqdm(imagelist):
    #     if counter ==2:
    #         pdf.add_page()
    #         counter = 0
    #     if im_name.endswith ('png'):
            
            
    #         # print(im_name)
        
            
    #         pdf.image(im_name,x = 0,y = 50+ 120* counter,w = 210,h = 100)
    #         counter +=1
        

    
    
def subsample_data(k,name,n=5000):
    for i, K in k.items():
        print (name+i+ ' size = ', len(K))
        if len(K)>n:
        #     # random sample -much larger sample
            idx=np.random.choice(len(K), replace = False, size = n)
            # newK = K.iloc[[idx]]
            k[i]=K.iloc[idx]
            print ('           ',name+i+ ' new size = ', len(k[i]))
    return k

def subsample_k(K,kInd,dir_data, ind = 'ind',n=500):
    lenK=len(K)
    if len(K)<=n:
       print (f'original size: {lenK}, file unchanged')

    else:# len(K)>n:
        # index file does not exist
        if not os.path.exists(f"{dir_data}{kInd}_subsample_indexes.p"):
    #     # random sample -much larger sample
            # idx=np.random.choice(len(K), replace = False, size = n)
            idx = np.asarray(K[ind].copy())
            np.random.shuffle(idx)
            idx = idx[:n]
            pickle_dump(f"{kInd}_subsample_indexes", idx,dir_data)
            
            print (f'original size: {lenK}, new size: {idx.shape[0]} indexes saved to file')
        else: #load from index file
           idx = pickle_load(f"{kInd}_subsample_indexes",dir_data)
           print (f'original size: {lenK}, new size: {idx.shape[0]} indexes loaded from file')
        newIdx = [ K.index[K[ind]==i][0] for i in idx ]#else print(i) 
        # newIdx = [ K.index[K.ind==i][0] if (i in K.ind) else print(i) for i in idx ]#else print(i) 
        # newIdx = [ K.index[K.ind==i][0] if (i in K.ind) else np.nan for i in idx ]


        if len(newIdx) != n:
            print ('size error')
        K=K.loc[newIdx]
       
        # newIdx ==idx

        
    return K
# for i,j in enumerate(idx):
#     try:
#         c= K.iloc[idx[i]]
#     except:
#         print(idxi)

def createAppendDataset(k,namesAll,kInd,uncommonFeatures ):
    # create an append dict; every sample is without its uncommon features and downsampled 
    # remove from data
    
    appendDict ={}
    for i in kInd:
        K=k[i].copy()
        for f in uncommonFeatures:
            try:
                K = K.drop(columns=[f])
            except:
                pass
        # print (K.columns)
        appendDict[i ] = K

    names  = removeFeatures(namesAll.copy(),uncommonFeatures)
    # append data
    # names['NamesAll'] 
    # k_append1= pd.DataFrame(columns =NamesAll)
    k_append= pd.DataFrame(columns = names['NamesAll']+ ['samp','ind'])
    for i, K in appendDict.items():

        # K= subsample_k(K[names['NamesAll']].copy(),n)
        # K['by_sample'] = int(i)
        # k_append1 = k_append1.append(K.copy(), ignore_index=True)
        k_append = pd.concat([k_append,K.copy()], ignore_index=True,axis=0,)
        
    # names['samp'] = k_append['sample'].copy()
    # names['ind'] = k_append['Ind'].copy()

    # unitest
    arr=[]
    for i in kInd:
        i = i[:-1] if 'f' in  i else i
        i = i[:-1] if 'a' in  i else i
        

        check = k_append.loc[k_append[k_append['samp']==float(i)].index[0]].isna()
        check = list(check[check].index)
        arr += [col for col in check if col not in arr]
       
    if len (arr)>0:
        print(f'fields to remove from df: {arr}')

    # DROP NULL COLUMNS; verify that all columns are mutual otherwise drop
    print(k_append.dropna(axis=1, how='all')[namesAll['NamesAll'].copy()].columns ==namesAll['NamesAll'].copy())


    return k_append,names




def getValsCsv(dir_data,vars,lensize = 10,fname = '_params.csv',vals = [] ): 
    if len(vals)>0:
        val1,val2 = vals
    else:
        newvars = [var + (lensize-len(var))*' ' for var in vars]

        df = pd.read_csv(dir_data+fname, sep =',',comment='#').astype(str)
        for col in df.columns:
            df[col] = [var + (lensize-len(var))*' ' for var in df[col]]
        
    
        for val, field in zip(newvars,['var','alg','samp']):
            df = df[df[field]==val].copy().drop(field,axis = 1)
        val1,val2 = df.values.tolist()[0]
        val1,val2 = float(val1),int(float(val2) )
    print(val1,val2)
    return val1,val2

import json
def getJ(j,group_ind,address,args):
    # if thers an external program runnig script (saving a json file) - take it
    # otherwise take the input j
    fname = 'j.json'
    try:   
        with open(fname, 'r') as f: 
            j,group_ind,address,args =  json.load(f)
        os.remove(fname)
    except:
        pass
    print(f'current j = {j},group_ind = {group_ind}, add = {address}')
    return j,group_ind,address,args  
# fname = 'j.json'
# val =['1','2']
# with open(fname, 'w') as f:
#     json.dump(val, f)
# j,group_ind = getJ(j=2,group_ind=4)


# # 
# def load_adjusted_batch(j,k,dir_indexes):
#     # k = k[j].copy()
#     Ind, by_sample= pickle_load(f"{j}_adjusted_subsample_indexes",dir_indexes)
#     uniq = [int(i) if i != 4.1 else float(i)  for i in np.unique(by_sample)]
#     # appendDict ={}
    
#     k_append= pd.DataFrame()
#     for samp in uniq:
#         K=k[k['by_sample']==samp].copy()
#         samp_idx = Ind[by_sample ==samp]
        
#         kInd = [True if (i in samp_idx) else False for i in np.asarray(K.Ind)]
#         K = K[kInd]
#         k_append = pd.concat([k_append,K.copy()], ignore_index=True,axis=0,)
#     return k_append



def figSettings(fig,figname,settings):
    if settings is not None:
        dir,show,saveSVG = settings
    else: #none
        import os
        dir,show,saveSVG = os.getcwd(), True,False

    fig.savefig(dir+figname+'.png', format="png", bbox_inches="tight", pad_inches=0.2)
    if saveSVG:
        fig.savefig(dir+figname+'.svg', format="svg", bbox_inches="tight", pad_inches=0.2)
    if show:
        plt.show()
    else:
        plt.close(fig)