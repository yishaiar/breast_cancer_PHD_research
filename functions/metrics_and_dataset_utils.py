from sklearn.metrics import confusion_matrix,classification_report, accuracy_score,precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve,auc,RocCurveDisplay
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pandas import Series

from random import Random

# define a class of metrics to use for the classification
# save figures to a folder
class ClassificationMetrics:
    def __init__(self,target:Series,pred:Series):
        self.target = target
        self.pred = pred
        self.labels = sorted(self.target.unique())
        # accuarcy
        # format 2 decimals of the accuracy

        self.acc = round(accuracy_score(self.target, self.pred),2)

    def plot_confusion_matrix(self,title = 'Confusion matrix',xlabel = 'Predicted label',ylabel = 'True label'):
        
        cm = pd.DataFrame(confusion_matrix(self.target, self.pred,labels = self.labels))
        cm.index = self.labels
        cm.columns = self.labels

        fig = plt.figure(figsize=(5,5))
        sns.heatmap(cm, annot = True)
        plt.title('Confusion matrix')    
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        return fig
    def plot_roc_curve(self):
        # labels to numeric classes dict
        numeric_labels : dict = {j:i for i,j in enumerate( self.labels)}
        # transform labels to numeric classes using labels to numeric classes dict 
        fpr, tpr, _ = roc_curve(self.target.map(numeric_labels), self.pred.map(numeric_labels))
        fig = plt.figure(figsize=(3, 8))
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc(fpr, tpr),estimator_name='ROC curve').plot()
        plt.title('ROC curve')
        return fig
    def plot_classification_report(self):
        
        clf_report = classification_report(self.target, self.pred,  output_dict=True)
        # target_names=target_names,
        clf_report = pd.DataFrame(clf_report).T
        clf_report['support'] = clf_report['support'].astype(int)

        fig = plt.figure(figsize=(5,5))

        sns.heatmap(clf_report, annot = True,
                    vmin=clf_report.values[:len( self.labels),:3].ravel().min(),
                    vmax=clf_report.values[:len( self.labels),:3].ravel().max(),)
        plt.title('Classification report')
        return fig
    def plot_classification_metrics(self):
        '''
        not needed, the classification report is enough
        '''
        avg = 'macro'
        acc = self.acc
        prec =  precision_score(self.target, self.pred, average = avg)
        rec = recall_score(self.target, self.pred, average = avg)
        f1 = f1_score(self.target, self.pred, average = avg)
        results = pd.DataFrame([ acc, prec, rec, f1],index = ['accuracy','precision','recall','f1'],columns = ['score']).T
        fig = plt.figure(figsize=(5,1))
        sns.heatmap(results, annot = True)
        return fig
        
    def plot_all(self,add = ''):
        fig1 = self.plot_confusion_matrix()
        fig1.savefig(add +'confusion_matrix.png')
        
        fig2 = self.plot_classification_report()
        fig2.savefig(add +'classification_report.png')
        

        if len(self.labels) == 2:#only plot roc curve if there are 2 classes (binary classification)
            fig3 = self.plot_roc_curve()
            fig3.savefig(add +'roc_curve.png')
        # self.plot_classification_metrics()#not needed, the classification report is enough
        # return accuracy_score
        return self.acc
def subset_labels(X,y,random_state,LEN = None):
    uniq = sorted(y.unique())
    min_len = min([len(y[y==label]) for label in uniq])
    min_len = min_len if not LEN else min(min_len,LEN//len(uniq))
    idx = []
    for i in uniq:
        label_idx = list(y[y==i].index)
        Random(random_state).shuffle(label_idx)
        idx += label_idx[:min_len]
    return X.loc[idx],y.loc[idx]
def createDataset(df,labels,samples = None,train_samples = None,test_samples = None,random_state=42,LEN = 4000):
    '''
    create a dataset for training and testing - 2 methods:
    1. random split using train_test_split - split a balanced random subset 
    2. split using the train_samples and test_samples lists of the original sample numbers determins which sample is for training and which is for testing
    '''
    

    

    

    if not train_samples:# train samples are not provided - split randomly
        
        # take only samples that are with labels - take their class-balanced subset (random sample)  - split it into train test randomly
        # idx =[]
        # uniq = sorted(labels.unique())
        # for label in uniq:
        #     label_idx = list(labels[labels==label].index)
        #     Random(random_state).shuffle(label_idx)
        #     idx += label_idx[:LEN//len(uniq)]
        
        X_train, X_test, y_train, y_test = train_test_split( df.loc[labels.index], labels, stratify = labels,
                                                        test_size=0.2,random_state=random_state, shuffle=True)
        X_train,y_train = subset_labels(X_train,y_train,random_state)
        X_test,y_test = subset_labels(X_test,y_test,random_state)
        print('train-test data split using random split')
    else: 
        # only take the samples that are with labels
        df = df.loc[labels.index]
        samples = samples.loc[labels.index]
        
        # only take the samples that are in the train and test list  
        X_train, X_test, y_train, y_test = df.loc[samples.isin(train_samples)],df.loc[samples.isin(test_samples)],labels.loc[samples.isin(train_samples)],labels.loc[samples.isin(test_samples)]
        
        

        X_train,y_train = subset_labels(X_train,y_train,random_state,LEN)
        X_test,y_test = subset_labels(X_test,y_test,random_state,LEN)
        
            
        
        
        print('train-test data split using sample indexes: train-{},test-{}'.format(train_samples,test_samples))     
    # print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
    if not (X_train.index==y_train.index).all() or not (X_test.index==y_test.index).all():
        print('train-test data split error: index mismatch')
    print(f'X_train: {len(X_train)}, X_test: {len(X_test)}')
    
    

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    target = pd.Series(i + '_' for i in np.random.randint(0,2,1000).astype(str))#random labels for testing
    pred = pd.Series(i + '_' for i in np.random.randint(0,2,1000).astype(str))#random labels for testing   
    ClassificationMetrics(target,pred).plot_all()
    



