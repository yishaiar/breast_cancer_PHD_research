import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
import numpy as np
import scanpy as sc
import anndata
from matplotlib.pyplot import rc_context
import pandas as pd


from functions import *


def drawUMAP(X_2d, NamesAll,CAll,settings,title = '',Figname = '' ):
        
    for M in NamesAll:
        cc=CAll[M]#[mask]
        plt.figure(figsize=(6, 5))
        plt.scatter(X_2d[:,0],X_2d[:,1],s=2,
                    c=cc, cmap=plt.cm.seismic)

        plt.colorbar()

        plt.clim(cc.quantile(0.01),cc.quantile(0.99))
        plt.title(title+ " - "+M)
        
        figname = Figname + M
    
        dir,show,saveSVG = settings
        plt.savefig(dir+figname+'.png', format="png", bbox_inches="tight", pad_inches=0.2)
        if saveSVG:
            plt.savefig(dir+figname+'.svg', format="svg", bbox_inches="tight", pad_inches=0.2)
        if show:
            plt.show()
        else:
            plt.close()
            
def drawDbscan(X,labels,core_samples_mask,settings,title='',figname=''):
    # Black removed and is used for noise instead.
    plt.figure(figsize=(6, 5))
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),label = k,
                 markeredgecolor='k', markersize=14)
        
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    
    plt.legend(fontsize=15, title_fontsize='40')    
    # plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.title(title)
    
    dir,show,saveSVG = settings
    plt.savefig(dir+figname+'.png', format="png", bbox_inches="tight", pad_inches=0.2)
    if saveSVG:
        plt.savefig(dir+figname+'.svg', format="svg", bbox_inches="tight", pad_inches=0.2)
    if show:
        plt.show()
    else:
        plt.close()
    
    
              
            

    
def plot_hist(k,NamesAll,figures,settings,func = sns.kdeplot ,title = '',Figname = '' ):
    
         
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
        figname = Figname + M
        
        dir,show,saveSVG = settings
        plt.savefig(dir+figname+'.png', format="png", bbox_inches="tight", pad_inches=0.2)
        if saveSVG:
            plt.savefig(dir+figname+'.svg', format="svg", bbox_inches="tight", pad_inches=0.2)
        if show:
            plt.show()
        else:
            plt.close()
            
        
     

def HeatMap(K,names,settings,title = '',figname = '' ):
    dir,show,saveSVG = settings

    Mat=K[K.Clust!=-1].groupby(by='Clust').mean()[names]
    amin=Mat[names].min().min()
    amax=Mat[names].max().max()
    g=sns.clustermap(Mat[names].T,cmap=plt.cm.seismic,vmin=amin,vmax=amax,
                    # figsize=(10,20), annot_kws={"size":8}, center=0,
                    figsize=(6, 5), annot_kws={"size":8}, center=0,
                    annot=True, linewidths=1,linecolor='k',)
    g.ax_col_dendrogram.set_title(title) 
    
    
    dir,show,saveSVG = settings
    plt.savefig(dir+figname+'.png', format="png", bbox_inches="tight", pad_inches=0.2)
    if saveSVG:
        plt.savefig(dir+figname+'.svg', format="svg", bbox_inches="tight", pad_inches=0.2)
    if show:
        plt.show()
    else:
        plt.close()
   
    
    
def plotClusters(K,X_2d,labels,NamesAll,settings,title = '',figname = '' ):
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
    
    with rc_context({'figure.figsize': (6, 5)}):
        sc.pl.umap(K_ann, color='clust', add_outline=True, legend_loc='on data',
                legend_fontsize=16, legend_fontoutline=4,frameon=True,
                title=title, palette=['r','orange','yellow','b'],show=False,projection='2d',)
    
    
    dir,show,saveSVG = settings
    plt.savefig(dir+figname+'.png', format="png", bbox_inches="tight", pad_inches=0.2)
    if saveSVG:
        plt.savefig(dir+figname+'.svg', format="svg", bbox_inches="tight", pad_inches=0.2)
    if show:
        plt.show()
    else:
        plt.close()
    
    
    # with rc_context({'figure.figsize': (4, 4)}):
    #     sc.pl.umap(K_ann, color=NamesAll+['clust'],ncols=5,vmax='p99.9',vmin='p0.001',
    #        cmap=plt.cm.seismic,add_outline=True,show=False)
    # plt.savefig(dir+figname+'_T.png', format="png", bbox_inches="tight", pad_inches=0.2)
    
   




  
  
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
        

   
   
def MeanDist(data1,data2,Markers,settings,title='',figname = '',xsize = 20):
    
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
    
    dir,show,saveSVG = settings
    plt.savefig(dir+figname+'.png', format="png", bbox_inches="tight", pad_inches=0.2)
    if saveSVG:
        plt.savefig(dir+figname+'.svg', format="svg", bbox_inches="tight", pad_inches=0.2)
    if show:
        plt.show()
    else:
        plt.close()


def plotSplit(K,i,min_x,min_y,settings,Figname):
  plt.figure(figsize=(6, 5))
  plt.title(f'K{i} ; limit = {np.round(min_x[i],4)}')
  sns.kdeplot(K['CD45'])
  plt.scatter(min_x[i],min_y[i])
  
  figname = '/K'+i+ Figname
  
  dir,show,saveSVG = settings
  plt.savefig(dir+figname+'.png', format="png", bbox_inches="tight", pad_inches=0.2)
  if saveSVG:
      plt.savefig(dir+figname+'.svg', format="svg", bbox_inches="tight", pad_inches=0.2)
  if show:
      plt.show()
  else:
      plt.close()