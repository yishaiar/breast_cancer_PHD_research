from scipy.stats import norm
import numpy as np
import pandas as pd
from usefull_functions import *

def fit1(df1,df2):
    mu1 = df1.mean()
    sigma1 = df1.std()

    mu2 = df2.mean()
    sigma2 = df2.std()

    df1_fit = (df1/sigma1-mu1/sigma1 + mu2/sigma2)*sigma2
    return df1_fit
def fit_distributions(df1,df2,subsample= True):
    # normalize the histograms (by number of sample )
    # df1 *= (len(df2)/ len(df1))

    
    df_ = pd.DataFrame()
    for i in range(10):
        df_[['mu1','sigma1']] = df1.apply(lambda x: norm.fit(x), axis=0).T
        df_[['mu2','sigma2']] = df2.apply(lambda x: norm.fit(x), axis=0).T
        # df1_fit = df1.copy()
        df1_fit = (df1.copy()/df_['sigma1']-df_['mu1']/df_['sigma1'] + df_['mu2']/df_['sigma2'])*df_['sigma2']
        # no index drop - only remove negar=tiv vals
        df1_fit[df1_fit < 0] = 0

    ind = (df1_fit[df1_fit.columns]>=0).all(axis=1)   
    df1_fit=df1_fit[ind].copy().reset_index(drop = True)

    if subsample:
        df1_Shape = len(df1_fit)
        df2_Shape = len(df2)
        if df1_Shape>df2_Shape:
            df1_fit = subsample_k(df1_fit.copy(),df2_Shape).reset_index(drop = True)
        else:
            df2 = subsample_k(df2.copy(),df1_Shape).reset_index(drop = True)

    print (len(df1_fit))
    print (len (df2))
    # since no ind remove - no need to do equivalent operation to df2
    



    # pdf = histogram / float(np.sum(histogram))

    return df1_fit,df2


# k['4_new1'] = fit1(k['4'].copy(),k['4.1'].copy())
# k['4'] , k['4.1'] = fit2(k['4_old'].copy(),k['4.1_old'].copy()) - better
