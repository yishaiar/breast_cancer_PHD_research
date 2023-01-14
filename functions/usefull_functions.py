import os
import pickle 
from tqdm import tqdm
import numpy as np
    
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
    test1 = all(item in new_NamesAll for item in NamesAll)
    # print (test1)
    if not test1:
        for i in new_NamesAll:
            NamesAll.remove(i)
        
    # print(NamesAll)

    test2 = not (any(item in CellIden for item in EpiCols))
    return test1*test2


    
from os import listdir
from os.path import isfile, join
from fpdf import FPDF
import cairosvg

def svg2png(dir_plots,j = '0'):
    print(dir_plots)
        
    imagelist = [ join(dir_plots, f) for f in listdir(dir_plots) if isfile(join(dir_plots, f))]
    for im_name in imagelist:
        if im_name.endswith ('png'):
          imagelist.remove (im_name[:-4])
          print(im_name[:-4])
    for im_name in imagelist:
        if im_name.endswith ('svg'):
            new_name = im_name+'.png'
            cairosvg.svg2png(url=im_name, write_to=new_name)
            



# cairosvg.svg2png(url=im_name, write_to=im_name+'.png')

def imList2pdf(dir_plots,j):
        
    print(dir_plots)
    
    imagelist = [ join(dir_plots, f) for f in listdir(dir_plots) if isfile(join(dir_plots, f))]
    imagelist.sort()
    pdf = FPDF()
    pdf.add_page()
    counter = 0
    
    for im_name in tqdm(imagelist):
        if counter ==2:
            pdf.add_page()
            counter = 0
        if im_name.endswith ('png'):
            
            
            # print(im_name)
        
            
            pdf.image(im_name,x = 0,y = 50+ 120* counter,w = 210,h = 100)
            counter +=1
        
            
    pdf.output(dir_plots+j+'.pdf', "F")
    
    
def subsample_data(k,name,n=5000):
    for i, K in k.items():
        print (name+i+ ' size = ', len(K))
        if len(K)>5000:
        #     # random sample -much larger sample
            idx=np.random.choice(len(K), replace = False, size = 5000)
            # newK = K.iloc[[idx]]
            k[i]=K.iloc[idx]
            print ('           ',name+i+ ' new size = ', len(k[i]))
    return k

def subsample_k(K,n=5000):
    print (' size = ', len(K))
    if len(K)>n:
    #     # random sample -much larger sample
        idx=np.random.choice(len(K), replace = False, size = n)
        # newK = K.iloc[[idx]]
        K=K.iloc[idx]
        print ('new size = ', len(K))
    return K


