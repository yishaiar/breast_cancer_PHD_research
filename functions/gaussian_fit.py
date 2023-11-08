import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

def bimodal_minima(x,y, maxfev=5000,plot=False):
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
    x0 = x[lim[0]:lim[1]][c]
    print (f'cutoff is at {x0}')

    if plot:
        plt.figure()
        plt.plot(x,bimodal(x,*params),color='red',lw=3,label='bimodal')
        plt.plot(x,d1,color='blue',lw=3,label='gausian1')
        plt.plot(x,d2,color='green',lw=3,label='gausian2')
        # # plt.scatter(x0,yy[c],color='red')
        plt.axvline(x=x0,label='minima',color='red')
        plt.plot(x,y,color='black',lw=1,label='data')

    return x0
# orig_params=(1,.2,300,2,.2,125)#[mean1,std1,A1,mean2,std2,A2]
# x,y,_,_ = gausianHistData(orig_params)
# x0 = bimodal_minima(x,y,plot=True)
