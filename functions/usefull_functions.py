import os
import pickle 
from tqdm import tqdm
import numpy as np
import pandas as pd
    
def deleteVars(to_delete=[]):
  for _var in to_delete:
      if _var in locals() or _var in globals():
          exec(f'del {_var}')
          
          
def removeFeatures(dict,features =['']):
  for key in dict:
    list = dict[key].copy()
    for feature in features:
      try:
        list.remove(feature)
      except:
        pass
    dict[key] = list
  return dict








def pickle_dump(file_name, dict,dir_data):
  with open(dir_data+file_name+'.p', "wb") as f:
    pickle.dump(dict, f, protocol=pickle.HIGHEST_PROTOCOL)


    
def pickle_load(file_name,dir_data):    
  with open(dir_data  + file_name + '.p', 'rb') as f:
      dict = pickle.load(f)
      print(file_name,'; loaded from file')
      return dict
  
  
def folderExists(path):
  if not os.path.exists(path):
   # Create a new directory because it does not exist
   os.makedirs(path)
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
from fpdf import FPDF




# cairosvg.svg2png(url=im_name, write_to=im_name+'.png')

def imList2pdf(dir_plots,j,groups):
        
    print(dir_plots)
    
    totalImagelist = [ join(dir_plots, f) for f in listdir(dir_plots) if isfile(join(dir_plots, f)) and f.endswith ('png')]
    for group in groups:
        name = j+'_'+group+'_'
        print(name)
        imagelist = [ f for f in totalImagelist if name in f]
        imagelist.sort()
        

        pdf = FPDF()
        pdf.add_page()

        for im_name in tqdm(imagelist):
            
            pdf.image(im_name,x = 0,y = 50,w = 210,h = 210)
            # counter +=1
            pdf.add_page()           
        pdf.output(dir_plots+name+'.pdf', "F")   
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

def subsample_k(K,kInd,subsample,dir_data,n=5000):
    lenK=len(K)
    if len(K)<=n:
       print (f'original size: {lenK}, file unchanged')

    else:# len(K)>n:
        if subsample:
    #     # random sample -much larger sample
            idx=np.random.choice(len(K), replace = False, size = n)
            # newK = K.iloc[[idx]]
            pickle_dump(f"{kInd}_subsample_indexes", idx,dir_data)
            
            print (f'original size: {lenK}, new size: {idx.shape[0]} indexes saved to file')
        else: #load from file
           idx = pickle_load(f"{kInd}_subsample_indexes",dir_data)
           print (f'original size: {lenK}, new size: {idx.shape[0]} indexes loaded from file')
        K=K.iloc[idx]
    return K


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
    names['NamesAll'] += ['by_sample','Ind']
    # k_append1= pd.DataFrame(columns =NamesAll)
    k_append= pd.DataFrame(columns = names['NamesAll'])

    for i, K in appendDict.items():

        # K= subsample_k(K[names['NamesAll']].copy(),n)
        # K['by_sample'] = int(i)
        # k_append1 = k_append1.append(K.copy(), ignore_index=True)
        k_append = pd.concat([k_append,K.copy()], ignore_index=True,axis=0,)
        
    names['by_sample'] = k_append['by_sample'].copy()
    names['Ind'] = k_append['Ind'].copy()

    # unitest
    arr=[]
    for i in kInd:
        check = k_append.loc[k_append[k_append['by_sample']==float(i)].index[0]].isna()
        check = list(check[check].index)
        arr += [col for col in check if col not in arr]
       
    if len (arr)>0:
        print(f'fields to remove from df: {arr}')
    return k_append,names




def getValsCsv(dir_data,vars,lensize = 10,fname = '_params.csv' ): 
    newvars = [var + (lensize-len(var))*' ' for var in vars]

    df = pd.read_csv(dir_data+fname, sep =',',comment='#').astype(str)
    for col in df.columns:
        df[col] = [var + (lensize-len(var))*' ' for var in df[col]]
    
 
    for val, field in zip(newvars,['var','alg','samp']):
        df = df[df[field]==val].copy().drop(field,axis = 1)
    val1,val2 = df.values.tolist()[0]
    # print(val1,val2 )
    return float(val1),int(float(val2) )
import json
def getJ(j,group_ind,address):
    # if thers an external program runnig script (saving a json file) - take it
    # otherwise take the input j
    fname = 'j.json'
    try:   
        with open(fname, 'r') as f: 
            j,group_ind,address =  json.load(f)
        os.remove(fname)
    except:
        pass
    print(f'current j = {j},group_ind = {group_ind}, add = {address}')
    return j,group_ind,address  
# fname = 'j.json'
# val =['1','2']
# with open(fname, 'w') as f:
#     json.dump(val, f)
# j,group_ind = getJ(j=2,group_ind=4)


# 
def load_adjusted_batch(j,k,dir_indexes):
    # k = k[j].copy()
    Ind, by_sample= pickle_load(f"{j}_adjusted_subsample_indexes",dir_indexes)
    uniq = [int(i) if i != 4.1 else float(i)  for i in np.unique(by_sample)]
    # appendDict ={}
    
    k_append= pd.DataFrame()
    for samp in uniq:
        K=k[k['by_sample']==samp].copy()
        samp_idx = Ind[by_sample ==samp]
        
        kInd = [True if (i in samp_idx) else False for i in np.asarray(K.Ind)]
        K = K[kInd]
        k_append = pd.concat([k_append,K.copy()], ignore_index=True,axis=0,)
    return k_append