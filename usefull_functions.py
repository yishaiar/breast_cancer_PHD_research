import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
import numpy as np
import scanpy as sc
import anndata
from matplotlib.pyplot import rc_context
import pandas as pd
import os

from functions import *


def plot_x2d(X_2d, NamesAll,CAll,dir,show = True,title = '',figname = '' ):
    if figname =='' or title =='':
        print('error - no fig file name')
    else:         
        for NN in NamesAll:
            Var=NN
            TSNEVar=NN
            cc=CAll[NN]#[mask]
            plt.figure(figsize=(6, 5))
            plt.scatter(X_2d[:,0],X_2d[:,1],s=2,
                        c=cc, cmap=plt.cm.seismic)

            plt.colorbar()

            plt.clim(cc.quantile(0.01),cc.quantile(0.99))
            plt.title(title+ " - "+TSNEVar)
            plt.savefig(dir+figname+NN+'.svg', format="svg", bbox_inches="tight", pad_inches=0.2)
            # plt.savefig(dir+figname+NN+'.png',dpi=200,bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close()
                

    
def plot_hist(k,NamesAll,figures,dir,show = True,func = sns.kdeplot ,title = '',figname = '' ):
    if figname =='' or title =='':
        print('error - no fig file name')
    else:         
        colors = cm.rainbow(np.linspace(0, 1, len (k.keys())))
        for M in NamesAll: 
            fig, ax = plt.subplots(1,2,figsize=(10,4))
            for [i, K],color,fig_num in zip(k.items(),colors,figures):
                fig_num -= 1
                
                # sns.kdeplot(K[M],c=c,label='Tumor ' + i)
                try: #if K doesnt contain the feature pass..
                  func(K[M],color=color,label='Tumor ' + i,ax = ax[fig_num])
                  # sns.kdeplot(K2[M],c='g',label='Tumor 2')
                  ax[fig_num].title.set_text(title)
                  ax[fig_num].legend()
                except:
                  pass

            
        
            
            plt.savefig(dir+figname+M+'.svg', format="svg", bbox_inches="tight", pad_inches=0.2)
            # plt.savefig('Plots/'+figname+M+'.png')
            if show:
                plt.show()
            else:
                plt.close()

def corrMat(K,names,dir,title = '',figname = '' ):

    Mat=K[K.Clust!=-1].groupby(by='Clust').mean()[names]
    amin=Mat[names].min().min()
    amax=Mat[names].max().max()
    g=sns.clustermap(Mat[names].T,cmap=plt.cm.seismic,vmin=amin,vmax=amax,
                    figsize=(10,20), annot_kws={"size":8}, center=0,
                    annot=True, linewidths=1,linecolor='k',)
    g.ax_col_dendrogram.set_title(title) 
    plt.savefig(dir+figname+'.svg', format="svg", bbox_inches="tight", pad_inches=0.2)
    # plt.savefig('Plots/'+figname+'.png')
   
    
    
def plotClusters(K,X_2d,labels,NamesAll,dir,title = '',figname = '' ):
    # bool if label frum cluster -1 (smallest cluster - maybe outlyer)
    m=labels!=-1
    # [NamesAll] - ALLOW TO CHANGE LIST OF FEATURES IN UMAP
    K_ann=anndata.AnnData(K[m][NamesAll],dtype=np.float32)
    K['Clust']=labels
    # # compute neighborhood graph used in umap - Added to K_ann
    sc.pp.neighbors(K_ann)
    # # embed the neighborhood graph using umap - Added to K_ann
    sc.tl.umap(K_ann,n_components=3)
    K_ann.obsm['X_umap']=X_2d[m]
    K_ann.obs['clust']=K[m].Clust.astype('category').values
    
    with rc_context({'figure.figsize': (5, 5)}):
        sc.pl.umap(K_ann, color='clust', add_outline=True, legend_loc='on data',
                legend_fontsize=16, legend_fontoutline=4,frameon=True,
                title=title, palette=['r','orange','yellow','b'],show=False,projection='2d',)
    plt.savefig(dir+figname+'.svg', format="svg", bbox_inches="tight", pad_inches=0.2)
    
    
    # with rc_context({'figure.figsize': (4, 4)}):
    #     sc.pl.umap(K_ann, color=NamesAll+['clust'],ncols=5,vmax='p99.9',vmin='p0.001',
    #        cmap=plt.cm.seismic,add_outline=True,show=False)
    # plt.savefig(dir+figname+'_T.png', format="png", bbox_inches="tight", pad_inches=0.2)
    
   

    
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


import pickle 
def pickle_dump(file_name, dict):
  with open('data/'+file_name+'.p', "wb") as f:
    pickle.dump(dict, f, protocol=pickle.HIGHEST_PROTOCOL)


    
def pickle_load(file_name):    
  with open('data/' + file_name + '.p', 'rb') as f:
      dict = pickle.load(f)
      print(file_name,'; loaded from file')
      return dict
    
    
# from os import listdir
# from os.path import isfile, join
# from fpdf import FPDF

# def imList2pdf(dir_plots):
#   imagelist = [ join(dir_plots, f) for f in listdir(dir_plots) if isfile(join(dir_plots, f))]
#   for im_name in imagelist:
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.image(im_name,x = 0,y = 50,w = 210,h = 100)
#     pdf.output(im_name+'.pdf', "F")


def umap_best(k,names,min_dist,n_neighbors):
    K = k.copy()
    CAll=pd.concat([K])
    # X_2d=draw_umap(CAll[CellIden+['p53']],cc=CAll['H4'],min_dist=0.05,n_neighbors=150,rstate=42)
    t= f'min_dist = {min_dist}, n_neighbors = {n_neighbors}'
    X_2d=draw_umap(CAll[names],cc=CAll['H4'],min_dist=min_dist,n_neighbors=n_neighbors,rstate=42,title=t)
    print(f'best settings: n_neighbors = {n_neighbors}')
    plt.show()
    
    return X_2d,CAll    
  
  
import itertools
def getList(valA,itersA):
  min_valA,max_valA = valA*2/3,valA*4/3
  A = np.round (min_valA + np.arange (itersA-1)*(max_valA- min_valA),4)
  A = np.insert(A,0,valA)
  return A

def umap_params(k,names ,dir_plots,
                valA, valB,
                itersA = 3, itersB = 3,
                j = 'testing'
                ):
    
  iters = list(itertools.product(getList(valA,itersA), getList(valB,itersB)))
  for i, [min_dist , n_neighbors] in enumerate(iters):      
      
    CAll=pd.concat([k.copy()])
    t= f'min_dist = {min_dist}, n_neighbors = {n_neighbors}'
    X_2d = calculate_umap(CAll[names],int(n_neighbors), min_dist)
    draw_umap (X_2d,CAll['H4'],dir_plots,
               show = True,
              title = t,
              figname = 'Tumor'+j+'_umap_CellIdentity')
    
    print(f'iteration:{i+1}/{len(iters)}, ' + t)
      
        


def dbscan_params(X_2d,dir_plots,
                  valA, valB,
                  itersA = 3, itersB = 3,
                  j = 'testing'
                  ):
    iters = list(itertools.product(getList(valA,itersA), getList(valB,itersB)))
    for i, [eps , min_samples] in enumerate(iters): 
        
      x_2d = X_2d.copy()
      t= f'eps = {eps}, min_samples = {min_samples}'
                  
      
      try:
        print(f'iteration:{i+1}/{len(iters)}, ' + t)
        X,labels,core_samples_mask = calculate_dbscan(data = x_2d,eps=eps,min_samples=int(min_samples))
        plot_dbscan(X,labels,core_samples_mask,dir_plots,show = True,
              title=t,
              figname='Tumor'+j+'_dbscan_CellIdentity'
              )
        print('\n')
      except: 
          print("error:")
      print(f'iteration:{i+1}/{len(iters)}, ' + t)
        
def folderExists(path):
  if not os.path.exists(path):
   # Create a new directory because it does not exist
   os.makedirs(path)
   print("The new directory is created!")
   
   
def MeanDist(data1,data2,Markers,dir,title='',figname = '',xsize = 20):
    
    sns.set_style({'legend.frameon':True})
 
    dd0=data1[Markers].mean().sort_values(ascending=False)
    dd1=data2[Markers].mean().sort_values()
    diffs=(dd1-dd0).sort_values(ascending=False)    
    
    clr=['darkgreen','purple']
    colors = [clr[0] if x < 0 else clr[1] for x in diffs]
    
    # fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
    plt.figure(figsize=(6, 5))
    plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=colors, alpha=1, linewidth=5)
    # Decorations
    plt.gca().set(ylabel='', xlabel='')
    plt.xticks(fontsize=xsize  ) 
    plt.yticks(fontsize=16 ) 
    # plt.xscale('symlog')
    

    plt.title(title, fontdict={'size':20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.savefig(dir+figname+'.svg', format="svg", bbox_inches="tight", pad_inches=0.2)
    # return fig
