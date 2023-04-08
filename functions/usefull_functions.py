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

def subsample_k(K,n=5000):
    lenK=len(K)
    if len(K)>n:
    #     # random sample -much larger sample
        idx=np.random.choice(len(K), replace = False, size = n)
        # newK = K.iloc[[idx]]
        K=K.iloc[idx]
    print (f'original size: {lenK}, new size: {len(K)}')
    return K


def getAppendDict(k,kInd,uncommonFeatures ):
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
    return appendDict


def createAppendDataset(names,appendDict,n):
    # del k; k=dict 

    # append data
    NamesAll = names['NamesAll']+['by_sample']
    k_append= pd.DataFrame(columns =NamesAll)
    for i, K in appendDict.items():

        K= subsample_k(K[NamesAll].copy(),n)
        # K['by_sample'] = int(i)
        k_append = k_append.append(K, ignore_index=True)
    by_sampleInd = k_append['by_sample'].copy()
    return k_append,by_sampleInd
    
    # k['1245'] = k_append

    # k['1245']['by_sample'] = by_sampleInd
    # k_append['by_sample'] = by_sampleInd
    # print(len(by_sampleInd))

def getValsCsv(dir_data,vars,lensize = 10,fname = 'params.csv' ): 
    df = pd.read_csv(dir_data+fname, sep =',',comment='#').astype(str)
    for col in df.columns:
        df[col] = [var + (lensize-len(var))*' ' for var in df[col]]
    newvars = []   
    for var in vars:
        newvars.append(var + (lensize-len(var))*' ')
    for val, field in zip(newvars,['var','alg','samp']):
        df = df[df[field]==val].copy().drop(field,axis = 1)
    val1,val2 = df.values.tolist()[0]
    # print(val1,val2 )
    return float(val1),int(float(val2) )
import json
def getJ(j,group_ind):
    # if thers an external program runnig script (saving a json file) - take it
    # otherwise take the input j
    fname = 'j.json'
    try:   
        with open(fname, 'r') as f: 
            j,group_ind =  json.load(f)
        os.remove(fname)
    except:
        pass
    print(f'current j = {j},group_ind = {group_ind}')
    return j,group_ind  
# fname = 'j.json'
# val =['1','2']
# with open(fname, 'w') as f:
#     json.dump(val, f)
# j,group_ind = getJ(j=2,group_ind=4)


# 
