import pickle
from matplotlib.pyplot import show,close
from os.path import exists 
from os import makedirs


# filter warnings
from warnings import filterwarnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=NumbaDeprecationWarning)
filterwarnings("ignore", category=NumbaPendingDeprecationWarning)


# from pandas import DataFrame,Series
# # from typing import List
# from numpy import ndarray
# from matplotlib.pyplot import figure


# set the print function of parent class to print into log file
# import  print_n_log  
# print = print_n_log.run('preproc', './preprocess.log', 'DEBUG')


class Parent(object):
        # def __getattr__(self, attr):#inherit from parent class (called when an attribute lookup has not found the attribute in the usual places)
    #     return getattr(self.self, attr)
    def get_attribute(self):
        """
        Prints all attribute of the self object (i.e configs)
        """

        for i in (vars(self)):
            print("{0:10}: {1}".format(i, vars(self)[i]))
            
    def pickle_dump(self,file_name, dict,dir_data):
        with open(dir_data+file_name+'.p', "wb") as f:
            pickle.dump(dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    def pickle_load(self,file_name,dir_data):    
        with open(dir_data  + file_name + '.p', 'rb') as f:
            dict_ = pickle.load(f)
            # dict_ = pd.read_pickle(dir_data  + file_name + '.p')
            print(file_name,'; loaded from file')
            return dict_
        
    def figSettings(self,fig,figname):

        # if self is None:#none
        #     import os
        #     settings = {'dir_plots':os.getcwd(), 'show':True, 'saveSVG':False}
        # else:pass
        format = 'png' if not self.saveSVG else 'svg'

        fig.savefig(self.dir_plots+figname+'.'+format, format=format, bbox_inches="tight", pad_inches=0.2)

        if self.show:
            show()
        else:
            close(fig)
    
    def folderExists(self,path,prompt = True):
        if not exists(path):
        # Create a new directory because it does not exist
            makedirs(path)
            print("The new directory is created!") if prompt else None
        print("The directory already exists!") if prompt else None
  
        
        
if __name__ == '__main__':
    # pass
    p = Parent()
    # p.get_attribute()
    # p.pickle_dump('test',{'a':1,'b':2})
    # p.pickle_load