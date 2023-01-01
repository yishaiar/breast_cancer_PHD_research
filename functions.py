import umap 


import numpy as np

import matplotlib.pyplot as plt


import seaborn as sns

from sklearn import metrics
from sklearn.cluster import  DBSCAN
from sklearn.preprocessing import StandardScaler
from lmfit import minimize, Parameters

import scanpy as sc







def wfall(shap_values, max_display=10, show=True):
    """ Plots an explantion of a single prediction as a waterfall plot.
    The SHAP value of a feature represents the impact of the evidence provided by that feature on the model's
    output. The waterfall plot is designed to visually display how the SHAP values (evidence) of each feature
    move the model output from our prior expectation under the background data distribution, to the final model
    prediction given the evidence of all the features. Features are sorted by the magnitude of their SHAP values
    with the smallest magnitude features grouped together at the bottom of the plot when the number of features
    in the models exceeds the max_display parameter.
    
    Parameters
    ----------
    shap_values : Explanation
        A one-dimensional Explanation object that contains the feature values and SHAP values to plot.
    max_display : str
        The maximum number of features to plot.
    show : bool
        Whether matplotlib.pyplot.show() is called before returning. Setting this to False allows the plot
        to be customized further after it has been created.
    """
    dark_o= mpl.colors.to_rgb('dimgray')
    dim_g= mpl.colors.to_rgb('darkorange')

    base_values = shap_values.base_values
    
    features = shap_values.data
    feature_names = shap_values.feature_names
    lower_bounds = getattr(shap_values, "lower_bounds", None)
    upper_bounds = getattr(shap_values, "upper_bounds", None)
    values = shap_values.values

    # make sure we only have a single output to explain
    if (type(base_values) == np.ndarray and len(base_values) > 0) or type(base_values) == list:
        raise Exception("waterfall_plot requires a scalar base_values of the model output as the first " \
                        "parameter, but you have passed an array as the first parameter! " \
                        "Try shap.waterfall_plot(explainer.base_values[0], values[0], X[0]) or " \
                        "for multi-output models try " \
                        "shap.waterfall_plot(explainer.base_values[0], values[0][0], X[0]).")

    # make sure we only have a single explanation to plot
    if len(values.shape) == 2:
        raise Exception("The waterfall_plot can currently only plot a single explanation but a matrix of explanations was passed!")
    
    # unwrap pandas series
    if safe_isinstance(features, "pandas.core.series.Series"):
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values

    # fallback feature names
    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(len(values))])
    
    # init variables we use for tracking the plot locations
    num_features = min(max_display, len(values))
    row_height = 0.5
    rng = range(num_features - 1, -1, -1)
    order = np.argsort(-np.abs(values))
    pos_lefts = []
    pos_inds = []
    pos_widths = []
    pos_low = []
    pos_high = []
    neg_lefts = []
    neg_inds = []
    neg_widths = []
    neg_low = []
    neg_high = []
    loc = base_values + values.sum()
    yticklabels = ["" for i in range(num_features + 1)]
    
    # size the plot based on how many features we are plotting
    pl.gcf().set_size_inches(8, num_features * row_height + 1.5)

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features == len(values):
        num_individual = num_features
    else:
        num_individual = num_features - 1

    # compute the locations of the individual features and plot the dashed connecting lines
    for i in range(num_individual):
        sval = values[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i])
            pos_widths.append(sval)
            if lower_bounds is not None:
                pos_low.append(lower_bounds[order[i]])
                pos_high.append(upper_bounds[order[i]])
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i])
            neg_widths.append(sval)
            if lower_bounds is not None:
                neg_low.append(lower_bounds[order[i]])
                neg_high.append(upper_bounds[order[i]])
            neg_lefts.append(loc)
        if num_individual != num_features or i + 4 < num_individual:
            pl.plot([loc, loc], [rng[i] -1 - 0.4, rng[i] + 0.4], color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
        if features is None:
            yticklabels[rng[i]] = feature_names[order[i]]
        else:
            yticklabels[rng[i]] = format_value(features[order[i]], "%0.03f") + " = " + feature_names[order[i]] 
    
    # add a last grouped feature to represent the impact of all the features we didn't show
    if num_features < len(values):
        yticklabels[0] = "%d other features" % (len(values) - num_features + 1)
        remaining_impact = base_values - loc
        if remaining_impact < 0:
            pos_inds.append(0)
            pos_widths.append(-remaining_impact)
            pos_lefts.append(loc + remaining_impact)
            c = dim_g  #colors.red_rgb
        else:
            neg_inds.append(0)
            neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)
            c = dark_o #colors.blue_rgb

    points = pos_lefts + list(np.array(pos_lefts) + np.array(pos_widths)) + neg_lefts + list(np.array(neg_lefts) + np.array(neg_widths))
    dataw = np.max(points) - np.min(points)
    
    # draw invisible bars just for sizing the axes
    label_padding = np.array([0.1*dataw if w < 1 else 0 for w in pos_widths])
    pl.barh(pos_inds, np.array(pos_widths) + label_padding + 0.02*dataw, left=np.array(pos_lefts) - 0.01*dataw, color=colors.red_rgb, alpha=0)
    label_padding = np.array([-0.1*dataw  if -w < 1 else 0 for w in neg_widths])
    pl.barh(neg_inds, np.array(neg_widths) + label_padding - 0.02*dataw, left=np.array(neg_lefts) + 0.01*dataw, color=colors.blue_rgb, alpha=0)
    
    # define variable we need for plotting the arrows
    head_length = 0.08
    bar_width = 0.8
    xlen = pl.xlim()[1] - pl.xlim()[0]
    fig = pl.gcf()
    ax = pl.gca()
    xticks = ax.get_xticks()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    bbox_to_xscale = xlen/width
    hl_scaled = bbox_to_xscale * head_length
    renderer = fig.canvas.get_renderer()
    
    # draw the positive arrows
    for i in range(len(pos_inds)):
        dist = pos_widths[i]
        arrow_obj = pl.arrow(
            pos_lefts[i], pos_inds[i], max(dist-hl_scaled, 0.000001), 0,
            head_length=min(dist, hl_scaled),
            color=dim_g, width=bar_width,
            head_width=bar_width
        )
        
        if pos_low is not None and i < len(pos_low):
            pl.errorbar(
                pos_lefts[i] + pos_widths[i], pos_inds[i], 
                xerr=np.array([[pos_widths[i] - pos_low[i]], [pos_high[i] - pos_widths[i]]]),
                ecolor=dim_g
            )

        txt_obj = pl.text(
            pos_lefts[i] + 0.5*dist, pos_inds[i], format_value(pos_widths[i], '%+0.02f'),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=12
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)
        
        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width: 
            txt_obj.remove()
            
            txt_obj = pl.text(
                pos_lefts[i] + (5/72)*bbox_to_xscale + dist, pos_inds[i], format_value(pos_widths[i], '%+0.02f'),
                horizontalalignment='left', verticalalignment='center', color=dim_g,
                fontsize=12
            )
    
    # draw the negative arrows
    for i in range(len(neg_inds)):
        dist = neg_widths[i]
        
        arrow_obj = pl.arrow(
            neg_lefts[i], neg_inds[i], -max(-dist-hl_scaled, 0.000001), 0,
            head_length=min(-dist, hl_scaled),
            color=dark_o, width=bar_width,
            head_width=bar_width
        )

        if neg_low is not None and i < len(neg_low):
            pl.errorbar(
                neg_lefts[i] + neg_widths[i], neg_inds[i], 
                xerr=np.array([[neg_widths[i] - neg_low[i]], [neg_high[i] - neg_widths[i]]]),
                ecolor=dark_o
            )
        
        txt_obj = pl.text(
            neg_lefts[i] + 0.5*dist, neg_inds[i], format_value(neg_widths[i], '%+0.02f'),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=12
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)
        
        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width: 
            txt_obj.remove()
            
            txt_obj = pl.text(
                neg_lefts[i] - (5/72)*bbox_to_xscale + dist, neg_inds[i], format_value(neg_widths[i], '%+0.02f'),
                horizontalalignment='right', verticalalignment='center', color=dark_o,
                fontsize=12
            )

    # draw the y-ticks twice, once in gray and then again with just the feature names in black
    ytick_pos = list(range(num_features)) + list(np.arange(num_features)+1e-8) # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    pl.yticks(ytick_pos, yticklabels[:-1] + [l.split('=')[-1] for l in yticklabels[:-1]], fontsize=13)
    
    # put horizontal lines for each feature row
    for i in range(num_features):
        pl.axhline(i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
    
    # mark the prior expected value and the model prediction
    pl.axvline(base_values, 0, 1/num_features, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
    fx = base_values + values.sum()
    pl.axvline(fx, 0, 1, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
    
    # clean up the main axis
    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().spines['left'].set_visible(False)
    ax.tick_params(labelsize=13)
    #pl.xlabel("\nModel output", fontsize=12)

    # draw the E[f(X)] tick mark
    xmin,xmax = ax.get_xlim()
    ax2=ax.twiny()
    ax2.set_xlim(xmin,xmax)
    ax2.set_xticks([base_values, base_values+1e-8]) # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax2.set_xticklabels(["\n$E[f(X)]$","\n$ = "+format_value(base_values, "%0.03f")+"$"], fontsize=12, ha="left")
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # draw the f(x) tick mark
    ax3=ax2.twiny()
    ax3.set_xlim(xmin,xmax)
    ax3.set_xticks([base_values + values.sum(), base_values + values.sum() + 1e-8]) # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax3.set_xticklabels(["$f(x)$","$ = "+format_value(fx, "%0.03f")+"$"], fontsize=12, ha="left")
    tick_labels = ax3.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform() + matplotlib.transforms.ScaledTranslation(-10/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform() + matplotlib.transforms.ScaledTranslation(12/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_color("#999999")
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    # adjust the position of the E[f(X)] = x.xx label
    tick_labels = ax2.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform() + matplotlib.transforms.ScaledTranslation(-20/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform() + matplotlib.transforms.ScaledTranslation(22/72., -1/72., fig.dpi_scale_trans))
    
    tick_labels[1].set_color("#999999")

    # color the y tick labels that have the feature values as gray
    # (these fall behind the black ones with just the feature name)
    tick_labels = ax.yaxis.get_majorticklabels()
    for i in range(num_features):
        tick_labels[i].set_color("#999999")
    
    if show:
        pl.show()

def calculate_dbscan(data,eps=0.1,min_samples=50):
    X=data
    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    try:
      print("Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(X, labels))
    except:
      print('Silhouette impossible; only 1 cluster recognized')
    return X,labels,core_samples_mask


    


    



# def dbscan_plot(data,eps=0.1,min_samples=50,title=''):
#     X=data
#     X = StandardScaler().fit_transform(X)
#     db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
#     core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#     core_samples_mask[db.core_sample_indices_] = True
#     labels = db.labels_

#     # Number of clusters in labels, ignoring noise if present.
#     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#     n_noise_ = list(labels).count(-1)

#     print('Estimated number of clusters: %d' % n_clusters_)
#     print('Estimated number of noise points: %d' % n_noise_)
#     print("Silhouette Coefficient: %0.3f"
#           % metrics.silhouette_score(X, labels))

#     # Black removed and is used for noise instead.
#     fig = plt.figure(figsize=(10, 10))
#     unique_labels = set(labels)
#     colors = [plt.cm.Spectral(each)
#               for each in np.linspace(0, 1, len(unique_labels))]
#     for k, col in zip(unique_labels, colors):
#         if k == -1:
#             # Black used for noise.
#             col = [0, 0, 0, 1]

#         class_member_mask = (labels == k)
        
#         xy = X[class_member_mask & core_samples_mask]
#         plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),label = k,
#                  markeredgecolor='k', markersize=14)
        
#         xy = X[class_member_mask & ~core_samples_mask]
#         plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#                  markeredgecolor='k', markersize=6)
    
#     plt.legend(fontsize=15, title_fontsize='40')    
#     # plt.title('Estimated number of clusters: %d' % n_clusters_)
#     plt.title(title)
    
    
    
#     print(f' eps = {eps},min_samples = {min_samples}')

#     return labels, fig


# labels,fig=dbscan_plot(data = X_2d,eps=eps,min_samples=min_samples,title=title)
# X_2d = calculate_umap(CAll[CellIden],n_neighbors, min_dist)
# draw_umap (X_2d,CAll['H4'],dir_plots,show = True,title=title,figname = figname)

def residual(params, x, data):
    alpha = params['alpha']
    beta = params['beta']
    gam = params['gamma']
 
 
    avMarkers=x['H3.3']*alpha+x['H4']*beta+x['H3']*gam
    od=x.subtract(avMarkers,axis=0)
    return np.std(od['H3.3'])+np.std(od['H4'])+np.std(od['H3'])


def residual2(params, x, data):
    beta = params['beta']
    gam = params['gamma']
 
 
    avMarkers=x['H4']*beta+x['H3.3']*gam
    od=x.subtract(avMarkers,axis=0)
    return np.std(od['H4'])+np.std(od['H3.3'])



def twoSampZ(X1, X2):
    from numpy import sqrt, abs, round
    from scipy.stats import norm
    mudiff=np.mean(X1)-np.mean(X2)
    sd1=np.std(X1)
    sd2=np.std(X2)
    n1=len(X1)
    n2=len(X2)
    pooledSE = sqrt(sd1**2/n1 + sd2**2/n2)
    z = ((X1 - X2) - mudiff)/pooledSE
    pval = 2*(1 - norm.cdf(abs(z)))
    return round(pval, 4)

def statistic(dframe):
    return dframe.corr().loc[Var1,Var2]


# def draw_umap(data,n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''
#               ,cc=0,rstate=42,dens=False):
#     fit = umap.UMAP(
#         n_neighbors=n_neighbors,
#         min_dist=min_dist,
#         n_components=n_components,
#         metric=metric, random_state=rstate, verbose=True, densmap=dens
#     )
#     u = fit.fit_transform(data);
#     fig = plt.figure(figsize=(6, 5))

#     if n_components == 2:
#         plt.scatter(u[:,0], u[:,1], c=cc,s=3,cmap=plt.cm.seismic)
#         plt.clim(-5,5)
#         plt.colorbar()
#     plt.title(title, fontsize=18)
    
    
#     return u, fig


def calculate_umap(data,n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', rstate=42,dens=False):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric, random_state=rstate, verbose=True, densmap=dens
    )
    u = fit.fit_transform(data)
    return u

    

def NormMark(data):
    params = Parameters()
    params.add('beta', value=0.1, min=0)
    params.add('gamma', value=0.1, min=0)
    params.add('alpha', value=0.1, min=0)
    ddf=data.copy()
    ddf2=data.copy()
    out = minimize(residual, params, args=(ddf, ddf),method='cg')
    beta=out.params['beta'].value
    gam=out.params['gamma'].value
    alpha=out.params['alpha'].value
    avMarkers=ddf['H3.3']*alpha+ddf['H4']*beta+ddf['H3']*gam
    ddf=ddf.subtract(avMarkers,axis=0)
    data=ddf
    ddf2[EpiCols]=data[EpiCols]
#    BCKData[NamesAll]=data[NamesAll]
    data=ddf2.copy()
    del ddf
    del ddf2
    return data

def NormMark2(data):
    params = Parameters()
    params.add('beta', value=0.1, min=-1000)
    params.add('gamma', value=0.1, min=-1000)

    ddf=data.copy()
    ddf2=data.copy()
    out = minimize(residual2, params, args=(ddf, ddf),method='cg')
    beta=out.params['beta'].value
    gam=out.params['gamma'].value

    avMarkers=ddf['H4']*beta+ddf['H3.3']*gam
    ddf=ddf.subtract(avMarkers,axis=0)
    data=ddf
    ddf2[EpiCols_M]=data[EpiCols_M]
#    BCKData[NamesAll]=data[NamesAll]
    data=ddf2.copy()
    del ddf
    del ddf2
    return data






def f(): raise Exception("Found exit()")



def BPlots(data,NMS,xVar='type'):
    for NN in NMS:
        BoxVar=NN
        plt.figure(figsize=(3, 5))    
        ax = sns.boxplot(x=xVar, y=NN, data=data,showfliers=False,palette=['red','blue'])
        plt.title(NN+" MGG")
        plt.show()   

def VPlots(data,NMS,xVar='type'):
    for NN in NMS:
        BoxVar=NN
        plt.figure(figsize=(3, 5))    
        ax = sns.violinplot(x=xVar, y=NN, data=data,showfliers=False,palette=['red','blue'])
        plt.title(NN+" MGG")
        plt.show()   


def KPlots(data,NMS,titleSup=''):
    for NN in NMS:
        plt.figure(figsize=(10,10))
        sns.kdeplot(data=data,x=NN,color='blue')
        
#        plt.legend()
        plt.title(""+NN+" "+titleSup)
        plt.show()





    
# def MedDist(data1,data2,Markers,title='',clr=['darkgreen','purple']):
#     sns.set_style({'legend.frameon':True})
 
#     dd0=data1[Markers].median().sort_values(ascending=False)
#     dd1=data2[Markers].median().sort_values()
#     diffs=(dd1-dd0).sort_values(ascending=False)    

#     colors = [clr[0] if x < 0 else clr[1] for x in diffs]
    
#     fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
#     plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=colors, alpha=1, linewidth=5)
#     # Decorations
#     plt.gca().set(ylabel='', xlabel='')
#     plt.xticks(fontsize=20 ) 
#     plt.yticks(fontsize=16 ) 

#     plt.title(title, fontdict={'size':20})
#     plt.grid(linestyle='--', alpha=0.5)    
    
def MeanDistIdU(data1,data2,Markers,title=''):
    sns.set_style({'legend.frameon':True})
 
    dd0=data1[Markers].mean().sort_values(ascending=False)
    dd1=data2[Markers].mean().sort_values()
    diffs=(dd1-dd0).sort_values(ascending=False)    
    colors = ['dodgerblue' if x < 0 else 'darkmagenta' for x in diffs]
    
    fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
    plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=colors, alpha=1, linewidth=5)
    # Decorations
    plt.gca().set(ylabel='', xlabel='')
    plt.xticks(fontsize=20 ) 
    plt.yticks(fontsize=16 ) 

    plt.title(title, fontdict={'size':20})
    plt.grid(linestyle='--', alpha=0.5)

def KPlot_Mrk(Mark,titleSup=''):
    plt.figure(figsize=(10,10))
    sns.kdeplot(data=C01,x=Mark,label="C01")
    sns.kdeplot(data=C02,x=Mark,label="C02")
    sns.kdeplot(data=C03,x=Mark,label="C03")
    sns.kdeplot(data=C04,x=Mark,label="C04")
    sns.kdeplot(data=C05,x=Mark,label="C05")
    plt.legend()
    plt.title(""+Mark+" "+titleSup)
    plt.show()
    
    
    
    

def UMAP_Plot(data1,data2,Markers,Set1='C01',Set2='Other',titleSup=''):
    data1=data1.assign(Set=Set1)
    data2=data2.assign(Set=Set2)
    CAll=data1.append(data2).sample(frac=0.1).copy()
    print(CAll)
    X_2d=draw_umap(CAll[Markers],cc=CAll['H3'],min_dist=0.01)
    for NN in NamesAll:
        cc=CAll[NN]#[mask]
        plt.figure(figsize=(6, 5))
        plt.scatter(X_2d[:,0],X_2d[:,1],s=2,
                    c=cc, cmap=plt.cm.jet)
    #    cmap = matplotlib.cm.get_cmap('jet')
        plt.colorbar()
    #    plt.clim(-3.5,3.5)
        plt.clim(cc.quantile(0.01),cc.quantile(0.99))
    #    mask=CAllmask[TSNEVar]==True
    #    rgba = cmap(-10)
    #    plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],s=2,
    #                color=rgba) 
        plt.title(NN+" "+titleSup)
        plt.show()

    plt.figure(figsize=(6, 5))
    mask=CAll.Set==Set1
    plt.scatter(X_2d[mask,0],X_2d[mask,1],s=2,
            c='blue', label=Set1)        
    mask=CAll.Set==Set2
    plt.scatter(X_2d[mask,0],X_2d[mask,1],s=2,
            c='red', label=Set2)        
    plt.legend()
    plt.show()
       

def DeltaCorr(data1,data2,Markers,titleSup=''):
    params = {'axes.titlesize': 30,
              'legend.fontsize': 20,
              'figure.figsize': (16, 10),
              'axes.labelsize': 20,
              'axes.titlesize': 20,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'figure.titlesize': 30}
    plt.rcParams.update(params)
    plt.style.use('seaborn-whitegrid')
    sns.set_style("white")

    print(titleSup)
    plt.figure(figsize=(20,20))
    matrix=data2[Markers].corr()-data1[Markers].corr()
    g=sns.clustermap(matrix, annot=True, annot_kws={"size":8},
                     cmap=plt.cm.jet,vmin=matrix.min().min(),vmax=matrix.max().max(),linewidths=.1); 
    plt.xticks(rotation=0); 
    plt.yticks(rotation=0); 

    plt.title(titleSup)
    plt.show()
    
    
def DefStyle():
    params = {'axes.titlesize': 30,
          'legend.fontsize': 20,
          'figure.figsize': (6, 5),
          'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'figure.titlesize': 30}
    plt.rcParams.update(params)
    plt.style.use('seaborn-whitegrid')
    sns.set_style("white")