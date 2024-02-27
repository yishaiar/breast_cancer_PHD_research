import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from usefull_functions import figSettings
from plot_functions import hex_to_rgba

# create data
def calcHist(p,bins):
    mean = p[0]
    std = np.abs(p[1])
    A = np.abs(p[2])
    n = int(np.round(A*std*bins))
    d = np.random.normal(mean,std,n)
    return d
def gausianHistData(params=None,bins = 100):
    
    d1 = calcHist(params[:3],bins)
    d2 = calcHist(params[3:],bins)
    # d1 = np.random.normal(1,.2,5000)
    # d2 = np.random.normal(2,.2,2500)
    data=concatenate((d1,d2))
    # y1,x1,_=hist(data,bins = 100,alpha=.3,label='data')
    y,x = np.histogram(data,bins=bins)

    x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
    
    return x,y,d1,d2

# ------------------------------------------------------------
# Define model function to be used to fit to the data above:
def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def bimodal_minima(x,y,manual_minima = None, maxfev=5000,settings = None):
    # estimate gausians with bimodal fit
    params,cov=curve_fit(bimodal,x,y,maxfev=maxfev)
    sigma = np.sqrt(np.diag(cov))
    # # get gaussians fit accordind to bimodal fit
    d1 = gauss(x,*params[:3])
    d2 = gauss(x,*params[3:])
    # # get limits according to gaussians maxima
    lim = np.sort([np.argmax(d1),np.argmax(d2)])

    # find index of minima betweeb maximas (bimodal minima)
    c = np.argmin(np.abs(d1[lim[0]:lim[1]]-d2[lim[0]:lim[1]]))
    x0 = x[lim[0]:lim[1]][c] if manual_minima is None else manual_minima
    print (f'cutoff is at {x0}')

    # if plot:

    fig,axs = plt.subplots(1,figsize=(6, 5))
    axs.set_title('bimodal cutoff (gassian fit)')

    axs.axvline(x=x0,label='minima',color='red')
    ind = x>x0
    x1,y1 = x[ind],y[ind]
    x2,y2 = x[~ind],y[~ind]
    # axs.plot(x,y,color='black',lw=1,label='data')
    c1,c2 = hex_to_rgba('3f78c1'),hex_to_rgba('b33d90')
    axs.plot(x1,y1,color=c1,lw=1,label='data')
    axs.plot(x2,y2,color=c2,lw=1,label='data')
    figSettings(fig,figname = 'bimodal_cutoff',settings = settings)

    axs.plot(x,d1,color='blue',lw=3,label='gausian1')
    axs.plot(x,d2,color='green',lw=3,label='gausian2')
    figSettings(fig,figname = 'bimodal_cutoff2',settings = settings)



    return x0 
# orig_params=(1,.2,300,2,.2,125)#[mean1,std1,A1,mean2,std2,A2]
# x,y,_,_ = gausianHistData(orig_params)
# x0 = bimodal_minima(x,y,plot=True)


# # -------------------------------
# if '.2a' in j and group_ind == 3:
#     labels = None# verify that the labels from previous are not used
#     if '_arcsinh' in j:
#         from gaussian_fit import *


#         x, y = sns.kdeplot(k['pRB'].copy(),label='label').lines[0].get_data()
#         y =  np.asarray(y)/.2 if '18.2' in j else  np.asarray(y)**.5 #highlight the gaussians diffrence
#         # plt.plot(x,y)

#         cutoff = bimodal_minima(x,y,maxfev = 5000,settings =settings)


#         labels = np.zeros(len(k)).astype(int)
#         labels[k['pRB']>cutoff] = 1

#         pickle_dump(f'Labels_{fname}', labels, dir_data)
        
#     try:labels = pickle_load(f'Labels_samp_arcsinh{j}_CellCycle_', dir_data) 
#     except:pass
#     names = ['pRB_low','pRB_high']
#     colors = draw_clusters(data = umapData.copy(),labels = labels,title='cutoff clusters',figname='cutoff_clusters',settings = settings,names =names)   


#     k['Clust'] = [names[l] for l in labels]#labels