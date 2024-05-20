import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import scanpy as sc
import anndata
# from matplotlib.pyplot import rc_context
import pandas as pd
import matplotlib.cm as cm
# import matplotlib.patches as mpatches
from usefull_functions import *
from functions import *
from csv import writer

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def backgroundColor_():
    return 'gainsboro' #gainsboro,lightgrey,whitesmoke

def drawUMAP1(X_2d, NamesAll,CAll,settings,title = '',Figname = '' ):
        
    for M in NamesAll:
        
        fig,axs = plt.subplots(1,figsize = (6, 5))
        cc=CAll[M]#[mask]


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





def drawUMAP(X_2d, NamesAll,CAll,settings,title = '',Figname = '',limits= [None,None,None,None] ):
    



    
        
    for M in NamesAll:
        
        
        cc=CAll[M]#[mask]
        drawSingleUMAP (X_2d, intensity = cc,name=M,settings = settings,limits = limits,title = title,Figname = Figname)

        # if M == 'Ki67':
        #     cc = np.log(cc)
        #     M = 'log_view_'+M
        #     drawSingleUMAP (X_2d, intensity = cc,name=M,settings = settings,title = title,Figname = Figname)
        
def drawSingleUMAP (X_2d, intensity,name,settings,title = '',Figname = '',limits = [None,None,None,None],backgroundColor = backgroundColor_() ):
        fig,axs = plt.subplots(1,figsize = (6, 5))
        if limits[0] is not  None:
            axs.set_xlim(limits[0], limits[2])  
            axs.set_ylim(limits[1], limits[3])
        else:
            axs.set_yticklabels([])
            axs.set_xticklabels([])
        
        axs.set_ylabel('umap2')
        axs.set_xlabel('umap1')

        axs.set_facecolor(backgroundColor)
        vmax=intensity.quantile(0.99);vmin=intensity.quantile(0.01)
        axs.scatter(X_2d[:,0],X_2d[:,1],s=2,c=intensity, cmap=plt.cm.seismic,
                    vmax=vmax,vmin=vmin) 
   
        norm = matplotlib.colors.Normalize(vmax=vmax,vmin=vmin)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=plt.cm.seismic),ax  = axs)
        axs.set_title(title+ " - "+name)        
        figname = Figname + name
        figSettings(fig,figname,settings)



def drawUMAPbySampleClust(X_2d, k,ind_cluster,M,settings,title = '',Figname = '',backgroundColor = backgroundColor_() ):
    

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
    plt.scatter(X_2d[:,0],X_2d[:,1],c = backgroundColor, alpha=0.2,s=2)
    plt.scatter(X_2d[ind_cluster][:,0],X_2d[ind_cluster][:,1],c = cc,s=2)
    # plt.legend([['0','1','2','3','4','5'])
    recs = []
    lgd=[]
    for i in range(0,arr.shape[0]):
        recs.append(matplotlib.patches.Rectangle((0,0),1,1,fc=colors[i]))
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
def drawUMAPbySample(X_2d,k, ind,labels,settings,colors = None,backgroundColor = backgroundColor_() ,title = '',Figname = '',):
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
    fig,axs = plt.subplots(1,figsize=(10, 10))
    # plt.figure(figsize=(10, 10))
    axs.scatter(X_2d['umap1'],X_2d['umap2'],c = backgroundColor, alpha=0.2,s=2)
    X_2d = X_2d.loc[ind].reset_index(drop=True)
    for u,cluster in zip(uniq,clusters):
        axs.scatter(X_2d.loc[cluster]['umap1'],X_2d.loc[cluster]['umap2'],c = cc[cluster],s=2,label = u,)#alpha=0.5
    
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
    
    axs.set_title(title)
    axs.legend(fontsize=15, title_fontsize='40',markerscale = 3.5,ncol=5,
        loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True,) #
    
    # figname = Figname
    figSettings(fig,Figname,settings)


    # dir,show,saveSVG = settings
    # plt.savefig(dir+figname+'.png', format="png", bbox_inches="tight", pad_inches=0.2)
    # if saveSVG:
    #     plt.savefig(dir+figname+'.svg', format="svg", bbox_inches="tight", pad_inches=0.2)
    # if show:
    #     plt.show()
    # else:
    #     plt.close()
def saveCsv(dir_plots,name,arr):
    with open(dir_plots+name+'.csv', 'a') as f:
        w= writer(f)
        for clust,data in arr:
            for samp ,p in data:
                row = [f'{name}: cluster {clust} sample {samp} = {p}']
                w.writerow(row)
                print(f'{name}: cluster {clust} sample {samp} = {p}') 
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
    arr = np.sort(np.unique(k_cluster[M]).astype(float))# return only existing values in the data (0% not included)
    if names is None:
        names = arr
    else:
        
        # take names from arr (only the existing names in the data), can work with clusters since arr vals are all integers (not samples)
        names = np.asarray(names)[arr.astype(int)]

    for i in range(0,arr.shape[0]):
        percentage = ClustFeaturePercentage(k_cluster,M,arr[i])
        percentage_arr.append([names[i],np.round(percentage,2)]) 
    return percentage_arr
def get_colors():

    hex = ['e6194b', '3cb44b', 'ffe119', '4363d8', 'f58231', '911eb4', '46f0f0', 'f032e6', 
        'bcf60c', 'fabebe', '008080', 'e6beff', '9a6324', 'fffac8', '800000', 'aaffc3',
            '808000', 'ffd8b1', '000075', '808080', ]
    # colors = np.asarray([hex_to_rgba(h) for h in hex])
    colors = [hex_to_rgba(h) for h in hex]
    return colors
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D
def drawDbscan(labels,dbData,settings,colors = None,title='',figname='',figsize=(6, 5)):
    X,core_samples_mask = dbData

    

    fig,axs = plt.subplots(1,figsize = figsize)
    axs.set_title(title)
    alpha = 1.0
    for label, color in zip(np.sort(np.unique(labels)), colors):
        class_member_mask = (labels == label)
        
        xy = X[class_member_mask & core_samples_mask]
        axs.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(color),
                markeredgecolor='k', 
                markersize=14,
                
                # alpha = alpha,  
                # add label for legend  
                label = label,)
        
        # cluster edges with smaller marker size for finess
        xy = X[class_member_mask & ~core_samples_mask]
        axs.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(color),
                 markeredgecolor='k', 
                 markersize=6,
                #  alpha = alpha,  
                 )
        


    # def update(handle, orig):
    #     handle.update_from(orig)
    #     handle.set_alpha(1)
        
    def update(handle, orig):
        handle.update_from(orig)
        handle.set_alpha(1)

    axs.legend(handler_map={PathCollection : HandlerPathCollection(update_func= update),
                            plt.Line2D : HandlerLine2D(update_func = update)})

    axs.legend(fontsize=15, title_fontsize='40',
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.05),
        # fancybox=True, 
        # shadow=False, 
        ncol=5,)
        # handler_map={PathCollection : HandlerPathCollection(update_func= update)},#update legend alpha
        # )



    figSettings(fig,figname,settings)
    # return colors
              
# def cluster_colors(labels):
#     unique_labels = np.sort(np.unique(labels))
#     cmap = get_colors()
#     if len(unique_labels)-1>cmap.shape[0]: #too many clusters for the preselected colors map
#     #cmap of Set2, twilight, PuOr and cividis.
#         cmap = [plt.get_cmap('Set2')(each) for each in np.linspace(0, 1, len(unique_labels)-1)]

#     # cmap = [plt.get_cmap('PuOr')(each) for each in np.linspace(0, 1, len(unique_labels)-1)]

#     colors = [(0, 0, 0, 1)]+[cmap[each] for each in range(len(unique_labels)-1)]# noise (-1 label) is black color
#     return colors           

    
import random   

def plot_hist(k,NamesAll,figures,settings,func = sns.kdeplot ,title = '',Figname = '',numSubplots = 4 ,colors=None):
    # pRB+ = 3f78c1; pRB- = b33d90
    if colors is None:
        colors = cm.rainbow(np.linspace(0, 1, len (figures.keys()))) 
        random.shuffle(colors)
    else: colors = [colors[i] for i,_ in enumerate(figures.keys())]
    for M in NamesAll: 
        fig, ax = plt.subplots(1,numSubplots,figsize=(int(4*numSubplots),4))
        # x1 =np.inf;x2 =-np.inf;y1 =np.inf;y2 =-np.inf;
        for i, color in zip(figures.keys(),colors):
            K = k[i]
            fig_num = figures[i] - 1

            try: #if K doesnt contain the feature pass..
              ax_ = ax if numSubplots==1 else ax[fig_num] 
              func(K[M],color=color,label='Tumor ' + i,ax = ax_)
              # sns.kdeplot(K2[M],c='g',label='Tumor 2')
              ax_.title.set_text(title)
              ax_.legend()
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
     

def HeatMap(K,labels,features,settings,labels_type=None,title = '',figname = '' ,csv = True,figure = True,amin = None,amax = None,rot_tentogram = False):
    Mat = K.groupby(by=labels).mean(numeric_only=True)[features]
    Mat = Mat.copy().loc[np.sort(Mat.index)]
    
    if labels_type == 'samp':
        Mat.drop([i for i in Mat.index if '.1'  in str(i)],inplace=True)



    if csv:
        Mat.to_csv(settings['dir_plots']+figname+'.csv')
    if figure:
        plotHeatMap(Mat,title,settings,figname,amin = amin,amax = amax,rot_tentogram = rot_tentogram)


def plotHeatMap(Mat,title,settings,figname,amin = None,amax = None,rot_tentogram = False):    #plot heatmap or clustermap or corrmap orr comfusion matrix
    figsize = (max(10, int(np.ceil(len(Mat.index)*0.7))), max(10, int(np.ceil(len(Mat.columns)*0.7))))
    # fig,axs = plt.subplots(1,figsize=(10, 10))


    amin=Mat.min().min() if amin is None else amin
    amax=Mat.max().max() if amax is None else amax
    if rot_tentogram:
        row_cluster,col_cluster=True,False,
    else:    
        row_cluster,col_cluster=False,True,

    # vmin,vmax - defines the  colormap dynamic range
    g = sns.clustermap(Mat.T,cmap=plt.cm.seismic,vmin=amin,vmax=amax,
                    row_cluster=row_cluster,col_cluster=col_cluster,
                    figsize=figsize, annot_kws={"size":8}, center=0,
                    annot=True, linewidths=1,linecolor='k')
    g.ax_col_dendrogram.set_title(title) 
    plt.show()
    figSettings(g,figname,settings)

   
    
    
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


def draw_clusters(data,labels,settings,names=[],title='',figname='',figsize=(6, 5),backgroundColor = backgroundColor_(),crossLabels = [None,None,None]):
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    if len(names)==0:
        names = unique_labels
    for i in unique_labels:#add probabiliy info
        p = np.sum(labels==i)/len(labels)
        n = f'{names[i]} ({np.round(p*100,2)}%)'
        names[i] = n
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, max(len(unique_labels),np.max(unique_labels)+1))]




 
    data = np.asarray(data)
    fig,axs = plt.subplots(1,figsize = figsize)
    

    # plt.figure(figsize = figsize)
    axs.set_title(title)

    for i ,color in zip(unique_labels,colors):
        ind = labels==i

        axs.plot(data[ind, 0],data[ind, 1], 'o', markerfacecolor=tuple(color),
        markeredgecolor='k', markersize=14,
        # add label for legend
        # label = names_[ind][0],
        label = names[i],)

        #
        # axs.scatter(data[ind, 0],data[ind, 1],c= cc[ind], label = names_[ind][0],)
                    # s=2,alpha = 0.5)#label = labels[ind]
    axs.legend(fontsize=15, title_fontsize='40',
        loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=2)
    axs.set_facecolor(backgroundColor)#xkcd:salmon
    
 
        



    figSettings(fig,figname,settings)
    return colors



def figSettings(fig,figname,config):

    # if config is None:#none
    #     import os
    #     settings = {'dir_plots':os.getcwd(), 'show':True, 'saveSVG':False}
    # else:pass
    format = 'png' if not config['saveSVG'] else 'svg'

    fig.savefig(config['dir_plots']+figname+'.'+format, format=format, bbox_inches="tight", pad_inches=0.2)

    if config['show']:
        plt.show()
    else:
        plt.close(fig)