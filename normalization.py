from lmfit import minimize, Parameters, report_fit
from tqdm import tqdm
import pandas as pd
import numpy as np
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
