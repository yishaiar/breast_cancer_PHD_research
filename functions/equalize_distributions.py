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

# ----------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import norm
import scipy.stats as stats

def qqPlot(data,refData):
    # Create sorted quantiles of the data and the corresponding theoretical quantiles
    # each data point gets a quantile in both the sorted quantiles,theoretical quantiles (refernce data)
    sorted_data = np.sort(data)
    theoretical_quantiles = np.percentile(refData, np.linspace(0, 100, len(refData)))

    # Plot the QQ plot
    plt.figure()
    plt.scatter(theoretical_quantiles, sorted_data ,marker = ".")
    plt.plot(theoretical_quantiles, theoretical_quantiles, color='r')
    plt.title('QQ Plot of Normal Distribution')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.show()



def qqFit(data1,data2,data3):
    data1_quantiles = np.percentile(data1, np.linspace(0, 100, len(data3)))
    data2_quantiles = np.percentile(data2, np.linspace(0, 100, len(data3)))
    # we assume the distribution of data1,data2 is the same  
    # changes are due to machne error (they are anchor to each other) 
    # create lookup table where for any quantile q2_i in data2  
    # f(q2_i)=q1_i:
    # data2_into_data1 =np.interp(data3, data2_quantiles, data1_quantiles)
    
    # data3 is not with the same distribution as data1,data2 but with same errors as data 2
    # i.e we transfer it to be with same errors as data1 to be able to compare to each other
    data3_into_data1 = np.interp(data3, data2_quantiles, data1_quantiles)
    return data3_into_data1

def LinearFit(data1,data2,data3):
    # we assume the distribution of data1,data2 is the same  
    # changes are due to machine error (they are anchor to each other) 
    # create lookup table where for any quantile q2_i in data2  

    
    # data3 is not with the same distribution as data1,data2 but with same errors as data 2
    # i.e we transfer it to be with same errors as data1 to be able to compare to each other
    mu1,sigma1 = norm.fit(data1)
    mu2,sigma2 = norm.fit(data2)

    # mu1 = np.mean(data1)
    # sigma1 = np.std(data1)
    # mu2 = np.mean(data2)
    # sigma2 = np.std(data2)

    data3_into_data1 = (data3/sigma2-mu2/sigma2 + mu1/sigma1)*sigma1
    return data3_into_data1

def MixedFit(data1,data2,data3):

    # we assume the distribution of data1,data2 is the same  
    # changes are due to machine error (they are anchor to each other) 
    # create lookup table where for any quantile q2_i in data2  
    # f(q2_i)=q1_i:
    # data2_into_data1 =np.interp(data3, data2_quantiles, data1_quantiles)
    
    # data3 is not with the same distribution as data1,data2 but with same errors as data 2
    # i.e we transfer it to be with same errors as data1 to be able to compare to each other
    data1 = np.array(data1)
    data2 = np.array(data2)
    data3 = np.array(data3)

    # data2_quantiles = np.percentile(data2, np.linspace(0, 100, len(data2)))
    # ind = (data3>np.max(data2_quantiles))*1 +(data3<np.min(data2_quantiles))*1
    ind = (data3>np.max(data2))*1 +(data3<np.min(data2))*1
    ind1 =[i for i in np.arange(data3.shape[0]) if ind[i]==1]
    ind2 =[i for i in np.arange(data3.shape[0]) if ind[i]==0]
    
    data3_into_data1 = np.zeros(data3.shape[0])

    data3_into_data1[ind1] = LinearFit(data1,data2,data3[ind1])
    data3_into_data1[ind2] = qqFit(data1,data2,data3[ind2])
    

    return data3_into_data1 

def fitDF(fit_into,fit_from,df_to_fit,func):
    cols = [col for col in fit_into.columns if col in fit_from.columns]
    dropped_cols = [col for col in df_to_fit.columns if col not in cols]
    # if len(dropped_cols)>0:
    #     print (f'cols dropped due to fitting: {dropped_cols} ')

    fitted_df = pd.DataFrame(columns=cols)
    for col in cols:
        fitted_df[col] = func(fit_into[col],fit_from[col],df_to_fit[col])
    return fitted_df,dropped_cols
# qqFit(data1,data2,data3[ind2])



# df = pd.DataFrame({'data1':np.random.normal(loc=0, scale=2, size=1000),
#                     'data2': np.random.normal(loc=100, scale=5, size=1000),
#                     'data3': np.random.normal(loc=100, scale=10, size=1000)})
                   

# # data1 = np.random.normal(loc=40, scale=5, size=1000)
# # data2 = np.random.normal(loc=5, scale=4, size=100)
# # data3 = np.random.normal(loc=7, scale=2, size=500)
# sns.kdeplot(df['data1'])
# sns.kdeplot(df['data2'])
# sns.kdeplot(df['data3'])




# # qqfit
# df['data3_into_data1_qq'] = qqFit(df['data1'],df['data2'],df['data3'])
# sns.kdeplot(df['data3_into_data1_qq'])
# # LinearFit
# df['data3_into_data1_LinearFit'] = LinearFit(df['data1'],df['data2'],df['data3'])
# sns.kdeplot(df['data3_into_data1_LinearFit'])

# # data3_into_data1_qq_drop - drop values outside range of data2
# # data3_into_data1_qqWithLinear - fit values outside range of data2 using linear


# df['data3_into_data1_qqWithLinear'],data3_into_data1_qq_drop = fit(df['data1'],df['data2'],df['data3'])
# sns.kdeplot(df['data3_into_data1_qqWithLinear'])


# sns.kdeplot(data3_into_data1_qq_drop)