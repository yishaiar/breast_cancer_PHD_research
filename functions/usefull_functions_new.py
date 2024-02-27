
from pandas import DataFrame,Series
from numpy import round,array
from numpy.random import choice

def removeFeatures(dict,remove_features =['']):
  
  for key in dict.keys():
    if dict[key] is not None:
        list_ = dict[key].copy()

        for feature in remove_features:
            try:
                list_.remove(feature)
            except:
                pass
        dict[key] = list_
  return dict




def cluster_samples_probability(labels:Series,data:Series,config:dict = None,sample_names:dict = None ,cluster_type_name:str= '' ,fname:str= ''):
    '''
    get in labeled clusters  the probability of each unique label in selected data (such as the samples id etc)
    labels: Series of the cluster labels
    data: Series of  data to check clustering probability
    names: dictionary of the names of the data (such as the samples id etc)
    config: dictionary of the configuration of the output file
    fname: the name of the output file
    '''
    
    p = []
    clusters = sorted(labels.unique())#[1:]# remove -1 (unclustered)
    for clustNum in clusters:
        cluster = f'{cluster_type_name} {clustNum}'
        single_cluster_samples = data.loc[labels[labels == clustNum].index].copy()
        LEN = len(single_cluster_samples)
        sample_percentage =[]
    
        for sample in sorted(single_cluster_samples.unique()):# return only existing samples in the cluster (0% not included)
            sample_ = sample if not sample_names else sample_names[sample]
            sample_percentage.append(f'{sample_} : {round(len(single_cluster_samples[single_cluster_samples==sample])/LEN*100,2)}%') 
        p+=[f'{cluster}']+sample_percentage
    # saveCsv(dir_plots, figname, arr)
    p = DataFrame(p)
    p.to_csv(config['dir_plots']+config['figname']+fname+'.csv')
    print('head of dataframe:')
    print(p.head())
    
    
def random_list(LEN = 1000,arr = ['Unknown', 'Noise' ,'Luminal' ,'Basal-like' ,'Cycling']):
    '''
    create random list of str's for testing purposes
    '''

    arr = choice(array(arr), [LEN])
    return ["".join(arr[i]) for i in range(len(arr))]