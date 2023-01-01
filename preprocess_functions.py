import numpy as np
from scipy import signal, stats
from lmfit import minimize, Parameters, report_fit
from tqdm import tqdm
import pandas as pd


def Gate(data,name,GateColumns):
    ddf=data.copy()
    print(name, 'gating on: ', GateColumns)
    print("Initial number of samples: ",len(ddf))
    # index of gating values ddf[['H3.3','H4']] larger than  Core gate value(5) 
    ind = (ddf[GateColumns]>5).all(axis=1)   
    ddf=ddf[ind]
    print("Core Gate: ",len(ddf))
    
    # index of quantile 99.99% gating values (remove outliers)
    quantile = np.quantile(ddf,0.9999,axis=0)
    outliers = ddf[(ddf>quantile).any(axis=1)]
    ddf=ddf[(ddf<quantile).all(axis=1)]
    print("Outlier Gate: ",len(ddf), ', samples removed:', len(outliers))
    # outliers distribution is not in a specific feature:
    
    # outliers [ outliers>quantile] = None
    # print(outliers.head(35))
    data=ddf.copy()
    del ddf
    return data

def arcsinh_transform(unscaled_data, scale=5):
    scaled_data=np.arcsinh(unscaled_data.copy()/scale)
    return scaled_data


# Kernel density estimation is the random variable's probability density function (PDF) estimatation
def splitInversePDF(K,i,feature, min_x = 'None'):
  
    # get data value for data with highest probability (peak)
      
  kernel = stats.gaussian_kde(K[feature])
  r=np.linspace(K[feature].min(),5,100)
  if min_x =='None':
    # avoid zero division using max; # inv_data = 1/np.maximum(1e-9,kernel(r))
    inv_data = 1/kernel(r)

    # minima : use builtin function fo find (min) peaks (use inversed data)
    # get index of input vector peaks, widths input - expected width of peaks of interest
    # min_peakind = signal.find_peaks_cwt(inv_data,np.arange(0.02,20,0.001))
    min_peakind = signal.find_peaks_cwt(inv_data,np.arange(1,100))

    min_x = r[min_peakind[0]]
  KCD45Neg = K[K[feature]<min_x].copy()
  min_y = kernel(min_x)[0]
  return KCD45Neg ,min_x, min_y





# Parameter - the quantity to be optimized; a value that can either be varied in the fit or held fixed value,
# can be constarint by lower, upper bounds or algebraic expression 

# minimize() - a wrapper around Minimizer for optimization problem for an objective function

def R(p,x,data,Q,M,M1,M2,M3):
    a=p['a']
    b=p['b']
    d=x.divide(a*M1+(1-a-b)*M2+b*M3,axis=0)
    return (d.std()['H3.3']+d.std()['H4']+d.std()['H3'])**2

def NormalizeNew(data,ToNorm):
    
    params = Parameters()
    params.add('a', value=0.1,min=0,max=1)
    params.add('b', value=0.1,min=0,max=1)
    ddf=data.copy()
    ddf2=data.copy()
    Q=ddf.mean()
    M=(ddf/Q)[['H3.3','H4','H3']].mean(axis=1)
    M1=(ddf/Q)['H3.3']
    M2=(ddf/Q)['H4']
    M3=(ddf/Q)['H3']

    out=minimize(R, params ,args=(ddf, ddf,Q,M,M1,M2,M3),method='cg')
    AA=out.params['a'].value
    BB=out.params['b'].value
    M=M1*AA+M2*(1-AA-BB)+M3*BB
    ddf=ddf.divide(M,axis=0).copy()
    data=ddf
    ddf2[ToNorm]=data[ToNorm]
    data=ddf2.copy()
    del ddf 
    del ddf2
    return data
    
def R2(params,      ddf, M1,M2):
    a=params['a']
    d=ddf.divide(a*M1+(1-a)*M2,axis=0)
    # var(x+y)= var(x)+var(y)+2*cov(x,y)
    return (d.std()['H3.3'])**2+(d.std()['H4'])**2

def NormalizeNew2(data,ToNorm):
    # normalization; division of each feature by feature mean over all samples  
    
    
    ddf=data.copy()
    ddf2=data.copy()
    Q=ddf.mean()
    M=(ddf/Q)[['H3.3','H4']].mean(axis=1)
    M1=(ddf/Q)['H3.3']
    M2=(ddf/Q)['H4']
    
    
    # find params value which minimizes function R2 output over args data 
    # optimization using method Conjugate-Gradient (cg) 
    # CG method is used to solve linear equation or optimize a quadratic equations; more efficient in those problems than gradient descent.
    # i.e find x which solves Ax=b
    params = Parameters()
    params.add('a', value=0.5,min=0.3,max=1)
    out=minimize(R2, params ,args=(ddf, M1,M2),method='cg')
    # print("# Fit using Conjugate-Gradient:\n")
    # report_fit(out)
    
    AA=out.params['a'].value

    M=M1*AA+M2*(1-AA)
    ddf=ddf.divide(M,axis=0).copy()
    data=ddf.copy()
    # print(data.shape,ddf2.shape)
    
    # ddf2[ToNorm] are normalized
    ddf2[ToNorm]=data[ToNorm]
    data=ddf2.copy()
    del ddf 
    del ddf2
    return data

def Mean_Core_normalization(K, ToNorm,coreFetures=['H3.3','H4']):
    Mean_Core=K[coreFetures].mean(axis=1)
    for N in ToNorm:
        K[N]=K[N]/Mean_Core
    return K

def scale_data(K):
    aaaa=pd.concat([K]).copy()
    m=np.mean(aaaa, axis=0)
    s=np.std(aaaa)
    K=(K-m)/s
    return K

