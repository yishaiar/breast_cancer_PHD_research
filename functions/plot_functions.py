import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import scanpy as sc
import anndata
from matplotlib.pyplot import rc_context
import pandas as pd
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from usefull_functions import *
from functions import *
from csv import writer



def drawUMAP1(X_2d, NamesAll,CAll,settings,title = '',Figname = '' ,log = False):
        
    for M in NamesAll:
        
        fig,axs = plt.subplots(1,figsize = (6, 5))
        cc=CAll[M]#[mask]
        if log:
            cc = np.log(cc)

        # axs.scatter(X_2d[:,0],X_2d[:,1],s=2,c=cc, cmap=plt.cm.seismic,)

        plt.scatter(X_2d[:,0],X_2d[:,1],s=2,
                    c=cc, cmap=plt.cm.seismic)
        # plt.colorbar(ax=axs)
        
        # norm = mpl.colors.Normalize(vmin=cc.quantile(0.01), vmax=cc.quantile(0.99))
        # fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=plt.cm.seismic),ax  = axs)
        plt.colorbar()
        # print(cc.quantile(0.01),cc.quantile(0.99))
        
        # axs.set_clim(cc.quantile(0.01),cc.quantile(0.99))

        plt.clim(cc.quantile(0.01),cc.quantile(0.99))
        # axs.set_title(title+ " - "+M)
        plt.title(title+ " - "+M)
        
        # figname = Figname + M
        # figSettings(fig,figname,settings)

        # dir,show,saveSVG = settings
        # plt.savefig(dir+figname+'.png', format="png", bbox_inches="tight", pad_inches=0.2)
        # if saveSVG:
        #     plt.savefig(dir+figname+'.svg', format="svg", bbox_inches="tight", pad_inches=0.2)
        # if show:
        #     plt.show()
        # else:
        #     plt.close()





def drawUMAP(X_2d, NamesAll,CAll,settings,title = '',Figname = '' ):
        
    for M in NamesAll:
        
        
        cc=CAll[M]#[mask]
        drawSingleUMAP (X_2d, intensity = cc,name=M,settings = settings,title = title,Figname = Figname)

        if M == 'Ki67':
            cc = np.log(cc)
            M = 'log_view_'+M
            drawSingleUMAP (X_2d, intensity = cc,name=M,settings = settings,title = title,Figname = Figname)
        
def drawSingleUMAP (X_2d, intensity,name,settings,title = '',Figname = '' ):
        fig,axs = plt.subplots(1,figsize = (6, 5))
        axs.set_facecolor('lightgray')#xkcd:salmon
        vmax=intensity.quantile(0.99);vmin=intensity.quantile(0.01)
        axs.scatter(X_2d[:,0],X_2d[:,1],s=2,c=intensity, cmap=plt.cm.seismic,
                    vmax=vmax,vmin=vmin)       
        norm = mpl.colors.Normalize(vmax=vmax,vmin=vmin)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=plt.cm.seismic),ax  = axs)
        axs.set_title(title+ " - "+name)        
        figname = Figname + name
        figSettings(fig,figname,settings)



def drawUMAPbySampleClust(X_2d, k,ind_cluster,M,settings,title = '',Figname = '' ):
    

    arr = np.unique(k[ind_cluster][M]).astype(int)
    # max_ = arr.shape[0]
    # colors1 = np.arange(max_)
    colors1 = np.arange(arr.max())
    arr1  = np.zeros(arr.max())
    ind=0
    for i in arr:
        arr1[int(i-1)] = int(colors1[ind])
        ind+=1
    
    # colors = cm.rainbow(colors1/(max_-1))
    colors = cm.rainbow(colors1/(arr.max()-1))

    
    cc = colors[(arr1[(k[ind_cluster][M]-1).astype(int)]).astype(int)]
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[:,0],X_2d[:,1],c = 'lightgrey', alpha=0.2,s=2)
    plt.scatter(X_2d[ind_cluster][:,0],X_2d[ind_cluster][:,1],c = cc,s=2)
    # plt.legend([['0','1','2','3','4','5'])
    recs = []
    lgd=[]
    for i in range(0,arr.shape[0]):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))
        percentage = ClustFeaturePercentage(k[ind_cluster],M,arr[i])
        lgd.append(f'{arr[i]} = {np.round(percentage,2)}%')
        # lgd.append(f'{arr[i]}')
    # plt.legend(recs,lgd,loc=4)
    plt.legend(recs,lgd,loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
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
def drawUMAPbySample(X_2d,k, ind,labels,settings,colors = None,backgroundColor = 'lightgrey' ,title = '',Figname = '',):
    # filter labels of data which is actually in the batch
    # todo..
    # labels = labels[1,:]
    # +1 due to existing clust value:-1
    if colors is None:
        cc = cm.rainbow((labels+1)/np.max(labels+1))
    else:
        # add black for -1 cluster
        cc = np.asarray([colors[l] for l in labels+1])

    clusters = []
    uniq = np.unique(labels)#uniq values(clusters in sample)
    for u in uniq:
        cluster = [i for i,j in enumerate(labels) if j==u ]
        clusters.append(cluster)
    # unique by_sample values
    # arr = np.unique(k[M]).astype(int)


    # max_ = arr.shape[0]
    # colors1 = np.arange(max_)
    # colors1 = np.arange(arr.max())
    # arr1  = np.zeros(arr.max())
    # ind=0
    # for i in arr:
    #     arr1[int(i-1)] = int(colors1[ind])
    #     ind+=1
    
    # colors = cm.rainbow(colors1/(max_-1))
    # colors1 = np.arange(arr.max())
    # colors = cm.rainbow(colors1/(arr.max()-1))

    # cc = colors[(arr1[(k[ind_cluster][M]-1).astype(int)]).astype(int)]
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[:,0],X_2d[:,1],c = backgroundColor, alpha=0.2,s=2)
    for u,cluster in zip(uniq,clusters):
        plt.scatter(X_2d[ind][cluster][:,0],X_2d[ind][cluster][:,1],c = cc[cluster],s=2,label = u,)#alpha=0.5
    
    # # plt.legend([['0','1','2','3','4','5'])
    # recs = []
    # lgd=[]
    # for i in range(0,arr.shape[0]):
    #     recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))
    #     percentage = ClustFeaturePercentage(k[ind_cluster],M,arr[i])
    #     lgd.append(f'{arr[i]} = {np.round(percentage,2)}%')
    #     # lgd.append(f'{arr[i]}')
    # # plt.legend(recs,lgd,loc=4)
    # plt.legend(recs,lgd,loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
    
    plt.title(title)
    plt.legend(fontsize=15, title_fontsize='40',markerscale = 3.5,ncol=5,
        loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True,) #
    
    figname = Figname

    dir,show,saveSVG = settings
    plt.savefig(dir+figname+'.png', format="png", bbox_inches="tight", pad_inches=0.2)
    if saveSVG:
        plt.savefig(dir+figname+'.svg', format="svg", bbox_inches="tight", pad_inches=0.2)
    if show:
        plt.show()
    else:
        plt.close()
def saveCsv(dir_plots,name,arr):
    with open(dir_plots+name+'.csv', 'a') as f:
        w= writer(f)
        for i,m in arr:
            for j ,k in m:
                row = [f'{name}: cluster {i} sample {j} = {k}']
                w.writerow(row)
                print(f'{name}: cluster {i} sample {j} = {k}') 
def ClustFeaturePercentage(cluster,feature,feature_val):
    # for f1 in np.unique(k[feature1]):
    #     print(f'{feature1} number: {f1}')
    #     cluster = k[k[feature1] == f1]
    #     clust_size = len(cluster)
    clust_size = len(cluster)
    sample = cluster[cluster[feature] == feature_val]
    percentage = len(sample)/clust_size*100
    return percentage     
def ClustPercentageBySample(k_cluster,M,names=None):
    percentage_arr =[]
    arr = np.unique(k_cluster[M]).astype(float)
    if names is None:
        names = arr
    for i in range(0,arr.shape[0]):
        percentage = ClustFeaturePercentage(k_cluster,M,arr[i])
        percentage_arr.append([names[i],np.round(percentage,2)]) 
    return percentage_arr
def drawDbscan(X,labels,core_samples_mask,settings,title='',figname='',figsize=(6, 5)):
    # Black removed and is used for noise instead.
    plt.figure(figsize = figsize)
    unique_labels = np.unique(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    # colors = [plt.cm.Spectral(each) 
    #           for each in np.linspace(0, 1, len(unique_labels))]
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
    
    # plt.legend(fontsize=15, title_fontsize='40') 
    plt.legend(fontsize=15, title_fontsize='40',
        loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
       
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
    
    return colors
              
            

    
import random   

def plot_hist(k,NamesAll,figures,settings,func = sns.kdeplot ,title = '',Figname = '',numSubplots = 4 ):
    
    colors = cm.rainbow(np.linspace(0, 1, len (figures.keys())))
    random.shuffle(colors)
    for M in NamesAll: 
        fig, ax = plt.subplots(1,numSubplots,figsize=(int(4*numSubplots),4))
        # x1 =np.inf;x2 =-np.inf;y1 =np.inf;y2 =-np.inf;
        for i, color in zip(figures.keys(),colors):
            K = k[i]
            fig_num = figures[i] - 1

            try: #if K doesnt contain the feature pass..
              func(K[M],color=color,label='Tumor ' + i,ax = ax[fig_num])
              # sns.kdeplot(K2[M],c='g',label='Tumor 2')
              ax[fig_num].title.set_text(title)
              ax[fig_num].legend()
            #   ------------------------
            #   ystart, yend = ax[fig_num].get_ylim()
            #   xstart, xend = ax[fig_num].get_xlim() 
            #   x1 = xstart if xstart<x1 else x1
            #   x2 = xend if xend>x2 else x2
            #   y1 = ystart if ystart<y1 else y1
            #   y2 = yend if yend>y2 else y2
            #  ------------------------- 
            except:
              pass
        # for fig_num in range(numSubplots):
        #     ax[fig_num].set_xlim(x1,x2)
        #     ax[fig_num].set_ylim(y1,y2)
        figSettings(fig,Figname + M,settings)




            
def scatter(k,f1,f2,name,figname,settings):
    plt.figure(figsize = (15,15))

    plt.scatter(k[f1],k[f2],marker = '.')
    plt.xlabel(f1)
    plt.ylabel(f2)

    plt.title(f'{f1} vs {f2}: {name}')
    figname = f'{figname}scatter_{f1}_{f2}'
    dir,show,saveSVG = settings
    plt.savefig(dir+figname+'.png', format="png", bbox_inches="tight", pad_inches=0.2)
    if saveSVG:
        plt.savefig(dir+figname+'.svg', format="svg", bbox_inches="tight", pad_inches=0.2)
    if show:
        plt.show()
    else:
        plt.close()
     

def HeatMap(k_clust,names,settings,clustFeature='Clust',
            title = '',figname = '' ):
    # dir,show,saveSVG = settings
    # k_clust = K[K.Clust!=-1]
    Mat=k_clust.groupby(by=clustFeature).mean()[names]
    # Mat = Mat[names]
    # csv
    Mat.to_csv(settings[0]+figname+'.csv')
    # heatmap figure
    matXsize = int(np.ceil(len(Mat.index)*0.7))
    matYsize = int(np.ceil(len(Mat.columns)*0.7))
    if matXsize<10:
        matXsize = 10
    if matYsize<10:
        matYsize = 10
    figsize = (matXsize, matYsize)
    
    plotHeatMap(Mat,title,settings,figname,figsize )

def plotHeatMap(Mat,title,settings,figname,figsize = (10, 10)):    
    amin=Mat.min().min()
    amax=Mat.max().max()

    # vmin,vmax - defines the  colormap dynamic range
    g=sns.clustermap(Mat.T,cmap=plt.cm.seismic,vmin=amin,vmax=amax,
                    # figsize=(10,20), annot_kws={"size":8}, center=0,
                    figsize=figsize, annot_kws={"size":8}, center=0,
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
   
    
    
def plotClusters(K,X_2d,labels,colors,NamesAll,settings,title = '',figname = '' ):
    # bool if label frum cluster -1 (smallest cluster - maybe outlyer)
    ind=labels!=-1
    # [NamesAll] - ALLOW TO CHANGE LIST OF FEATURES IN UMAP
    K_ann=anndata.AnnData(K[ind][NamesAll].copy(),dtype=np.float32)
    # K['Clust']=labels
    # # compute neighborhood graph used in umap - Added to K_ann
    sc.pp.neighbors(K_ann)
    # # embed the neighborhood graph using umap - Added to K_ann
    sc.tl.umap(K_ann,n_components=3)
    K_ann.obsm['X_umap']=X_2d[ind]
    K_ann.obs['clust']=K[ind].Clust.astype('category').values
    
    # with rc_context({'figure.figsize': (6, 5)}):
    # fig,axs = plt.subplots(1,figsize = (6, 5))
    fig = sc.pl.umap(K_ann, color='clust', add_outline=True, legend_loc='on data',
            legend_fontsize=16, legend_fontoutline=4,frameon=True,return_fig=True,
            title=title,show=False,projection='2d',palette=colors)#
    
    # palette=['r','orange','yellow','b']
    
    figSettings(fig,figname,settings)

    # return K_ann
   




  
  
import itertools
def getList(valA,itersA):
  min_valA,max_valA = valA*2/3,valA*4/3
  A = np.round (min_valA + np.arange (itersA-1)*(max_valA- min_valA),4)
  A = np.insert(A,0,valA)
  return A

# def umap_params(k,names ,dir_plots,
#                 valA, valB,
#                 itersA = 3, itersB = 3,
#                 j = 'testing'
#                 ):
    
#   iters = list(itertools.product(getList(valA,itersA), getList(valB,itersB)))
#   for i, [min_dist , n_neighbors] in enumerate(iters):      
      
#     CAll=pd.concat([k.copy()])
#     t= f'min_dist = {min_dist}, n_neighbors = {n_neighbors}'
#     X_2d = calculate_umap(CAll[names],int(n_neighbors), min_dist)
#     draw_umap (X_2d,CAll['H4'],dir_plots,
#                show = True,
#               title = t,
#               figname = 'Tumor'+j+'_umap_CellIdentity')
    
#     print(f'iteration:{i+1}/{len(iters)}, ' + t)
      
        


# def dbscan_params(X_2d,dir_plots,
#                   valA, valB,
#                   itersA = 3, itersB = 3,
#                   j = 'testing'
#                   ):
#     iters = list(itertools.product(getList(valA,itersA), getList(valB,itersB)))
#     for i, [eps , min_samples] in enumerate(iters): 
        
#       x_2d = X_2d.copy()
#       t= f'eps = {eps}, min_samples = {min_samples}'
                  
      
#       try:
#         print(f'iteration:{i+1}/{len(iters)}, ' + t)
#         X,labels,core_samples_mask = calculate_dbscan(data = x_2d,eps=eps,min_samples=int(min_samples))
#         plot_dbscan(X,labels,core_samples_mask,dir_plots,show = True,
#               title=t,
#               figname='Tumor'+j+'_dbscan_CellIdentity'
#               )
#         print('\n')
#       except: 
#           print("error:")
#       print(f'iteration:{i+1}/{len(iters)}, ' + t)
        

   
   
def MeanDist(data1,data2,Markers,settings,title='',figname = '',font_size = 10):
    
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
    plt.xticks(fontsize=font_size  ) 
    plt.yticks(fontsize=font_size ) 
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


def plotSplit(K,i,min_x,min_y,settings,neg_percentage,Figname,log = True,xsize = 2):
    #  plt.figure(figsize=(10, 5))
    fig, ax = plt.subplots(figsize=(2*xsize, xsize))
    sns.kdeplot(K,ax = ax)
    start, end = ax.get_xlim()
    res = 4/xsize
    ax.xaxis.set_ticks(np.arange(int(np.floor(start)), int(np.ceil(end)), res))
    # plt.xticks(np.arange(int(np.floor(K.min())),int(np.ceil(K.max())),0.2))
    

    # print(f'k{i} negative percentage: {neg_percentage}%')
    plt.title(f'K{i} ; limit = {np.round(min_x,4)}; neg percentage = {neg_percentage}%')

    
    plt.scatter(min_x,min_y)
    if log:
        plt.yscale('log')
    figname = '/K'+i+ Figname
    
    dir,show,saveSVG = settings
    plt.savefig(dir+figname+'.png', format="png", bbox_inches="tight", pad_inches=0.2)
    if saveSVG:
        plt.savefig(dir+figname+'.svg', format="svg", bbox_inches="tight", pad_inches=0.2)
    # if show:
    plt.show()
    # else:
    #     plt.close()


    #     # f, ax = plt.subplots(figsize=(6, 5))
    # # ax.set(xscale="log", yscale="log")
    # plt.yscale('log')
    # plt.figure(figsize=(6, 5))
    # plt.title(f'K{i} ; limit = {np.round(min_x[i],4)}')
    # sns.kdeplot(K['CD45'])
    # plt.scatter(min_x[i],min_y[i])

def saveCsv_split(dir_plots,name,arr):
    with open(dir_plots+name+'.csv', 'a') as f:
        w= writer(f)
        for m in arr:
            sample,feature,percentage = m
            row = [f'sample {sample}: {feature} percentage = {percentage}']
            w.writerow(row)
            # print(row)     

def createCorrMat(rawMat,method ='spearman',
                  settings =None ,title='',figname = ''):
    


    corrMatFeatures = rawMat.corr(method =method)
    corrMatSamples = rawMat.T.corr(method =method)

    
    if settings != None:
        # plotHeatMap(rawMat,'raw;'+title,settings,'raw_'+figname)
        rawMat.to_csv(settings[0]+'raw_corrmat_'+figname+'.csv')
        plotHeatMap(corrMatFeatures,'corrFeatures;'+title,settings,'corrFeatures_'+figname,
                    figsize = (20, 20))
        plotHeatMap(corrMatSamples,'corrSamples;'+title,settings,'corrSamples_'+figname)
        # plot matrices..

    # return kEpinuc,corrMatFeatures,corrMatSamples


def hex_to_rgba(hex):
    rgb = []
    for i in (0, 2, 4):
        decimal = int(hex[i:i+2], 16)
        # normalized rgb
        rgb.append(decimal/255)
    # alpha
    rgb.append(0.5)
    RGB = tuple(rgb)
    return RGB


def draw_kmeans(data,labels,settings,names=[],title='',figname='',figsize=(6, 5)):
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, max(len(unique_labels),np.max(unique_labels)+1))]

    # colors = cm.rainbow(np.linspace(0, 1, np.unique(labels).shape[0]))
    cc  = np.asarray([ colors[i] for i in labels])
    if len(names)==0:
        names = unique_labels

    names_ = np.empty(labels.shape,dtype=object)
    for i in unique_labels:
        p = np.sum(labels==i)/len(labels)
        n = f'{names[i]} ({np.round(p*100,2)}%)'
        names_[labels==i] = n
        

 
    data = np.asarray(data)
    fig,axs = plt.subplots(1,figsize = figsize)

    # plt.figure(figsize = figsize)
    axs.set_title(title)

    for i in unique_labels:
        ind = labels==i
        #
        axs.scatter(data[ind, 0],data[ind, 1],c= cc[ind], label = names_[ind][0])#label = labels[ind]
    axs.legend(fontsize=15, title_fontsize='40',
        loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=2)
    axs.set_facecolor('lightgray')#xkcd:salmon


    figSettings(fig,figname,settings)
    return colors
