#!/usr/bin/env python

"""
----------------------------------------------------
In this file few machine learning models are trained
to recognize human activity from the UCI Human Activity 
Recognition dataset
----------------------------------------------------
"""
import random
import csv
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

def Data_Split(df,IDs,personid,idcol,classif):
    """
    For a given 
    dataframe: df
    selected ID : IDs
    ID column name for IDs: personid
    ID column name in df : idcol
    classification column name : classif
    returns a array with feautres to be trained on : Xsub
    and array of classification label : Ysub
    """
    dfID = pd.DataFrame(IDs,columns = [personid])
    subset = dfID.merge(df,left_on = personid,right_on= idcol, how='left')
    subset = subset.dropna()
    Xsub = subset.drop(columns = [personid,idcol,classif]).values
    Xsub = scale(Xsub)
    
    Ysub = subset[classif].values.astype(int)
    return Xsub,Ysub
    
def Labels_to_num (df, classif,label_col):
    """
    For a dataframe (df) and its classification (classif) it 
    turns each class label to numerical values and returns the dataframe
    """
    
    ClassList= df[classif].unique()
    id = 0
    ClassDict = {}
    for ClassVal in ClassList:
        id = id+1
        ClassDict[ClassVal]=id
    print('Class Asssign ',ClassDict)
    for ClassNum in ClassDict.keys():
        df.ix[df[classif]==ClassNum,label_col] = ClassDict[ClassNum]
    df=df.drop(classif,axis =1)
        
    return df        
    
    
def ml(df,MODEL,person,weight_plot=0):
    """ Runs either logistic regression or KNN through sklearn using both 
        dev and test set. """
    print (MODEL)
    N_FOLDS =8
    ACC_THRESH = 0.01 # dev set accuracy must be x% better to use new param
    
    if MODEL == 'LOGISTIC':
        c_l = [0.01,0.03, 0.1, 0.3, 1,3,10,30]
    elif MODEL == 'KNN':       
        c_l = [25,20,18,15,10,8,9,8,7,6,3,2]
    elif MODEL == "DT":
        c_l =[1,2,3,4,5,6,7,8,9,10]
    elif MODEL == "SVC":
        c_l = [ 0.1, 0.3, 1,3,10,30,100,300,1000,3000,10000,0.01,0.03]
    regularizer = 'l1'
    
    # gets the unique id of the people in the dataset
    peopleid = df[person].unique()
    
    df_good = df.dropna()
    
    df_good = Labels_to_num(df_good,'Activity','y')
    skf = StratifiedKFold(shuffle = True, n_splits=N_FOLDS, random_state=7)

    acc_test_a = np.zeros(N_FOLDS)
    acc_train_a = np.zeros(N_FOLDS)
    
    X_nd = df_good.drop(['y',person],axis=1).values

    X_nd = scale(X_nd) # magnitude has useful info?         
    y_n = df_good['y'].values.astype(int)
    
    ncol = len(X_nd[0])
    weight_list = [0]* ncol
    for i, (train, test) in enumerate(skf.split(peopleid,np.zeros(len(peopleid)))):
        print(train.tolist())
        print(test.tolist())
        train_n = len(train)
        devID = train[:int(train_n/4)]  # empirically found that dev 1/4 is good
        sub_trainID = train[int(train_n/4):] # this is temporary train set
        
        Xdev,Ydev= Data_Split(df_good,devID,'personid',person,'y')
        Xsub,Ysub = Data_Split(df_good,sub_trainID,'personid',person,'y')
        
        best_acc = 0
        best_c = None
        # in this loop we find best hyper parameter for this split
        for c in c_l:
            if MODEL == 'LOGISTIC': 
                clf = linear_model.LogisticRegression(penalty=regularizer,C=c)
            elif MODEL == 'KNN':
               # if c<=(len(df_good.index)/2):
                clf = KNeighborsClassifier(n_neighbors=c, metric='euclidean',weights='uniform')
            elif MODEL == 'DT':
                clf = DecisionTreeClassifier (max_depth = c)
            elif MODEL == 'SVC':
                clf = SVC (C=c,kernel = 'rbf')
            
            clf.fit(Xsub, Ysub)
            y_pred = clf.predict(Xdev)
            acc = metrics.accuracy_score(y_pred,Ydev)
            if(acc > best_acc + ACC_THRESH*best_acc):
                best_acc = acc
                best_c = c

        # retrain with all train data and best_c
        print('fold:',i,' best c:',best_c, ' dev:%.2f' % best_acc) #' dev_ones:%.2f' % (y_n[dev].sum()/len(dev)),end='')
        if MODEL == 'LOGISTIC': 
            clf = linear_model.LogisticRegression(penalty=regularizer,C=best_c)
        elif MODEL == 'KNN':           
            clf = KNeighborsClassifier(n_neighbors=best_c, metric='euclidean',weights='uniform')
        elif MODEL == 'DT':
            clf = DecisionTreeClassifier(max_depth =best_c)
        elif MODEL == 'SVC':
            clf = SVC(C=best_c,kernel ='rbf')
            
        Xtrain, Ytrain = Data_Split(df_good,train,'personid',person,'y')
        Xtest, Ytest = Data_Split(df_good,test,'personid',person,'y')
        
        print('train size ',Xtrain.shape)
        print('test size ',Xtest.shape)
        clf.fit(Xtrain,Ytrain)
       
        #applicable for logisitic regression
        #Using dictionary to keep track of the number of times the weights were 0
        
        d = df_good.drop('y',axis=1)
        clust = list(d.columns.values)
        if weight_plot == 1:
            coeff=clf.coef_[0]
            weights_dict= dict(zip(clust,coeff))
            wn0= {}
            for k, v in weights_dict.items():
                if v == 0:
                    wn0[k] = v
            print ('  ',wn0)
            weight_list = [x+y for x,y in zip(coeff,weight_list)]
               
        y_predtest = clf.predict(Xtest)
        y_predtrain = clf.predict(Xtrain)
        
        # makes a confusion matrix
        
        Conf_Mat = confusion_matrix(Ytest, y_predtest,labels= [1,2,3,4,5,6])
        print(Conf_Mat)
        f1 = f1_score(Ytest,y_predtest,labels=None)
        
        acc_test_a[i] = metrics.accuracy_score(y_predtest,Ytest)
        acc_train_a[i] = metrics.accuracy_score(y_predtrain,Ytrain)
        print(' test:%.2f' % acc_test_a[i], ' train:%.2f' % acc_train_a[i])
        if i==0:
            break
                
    print('Avg test acc:%.3f' % acc_test_a.mean(),'Avg train acc:%.3f' % acc_train_a.mean())   
    
    # plotting the weights for logisitic regression
    if weight_plot == 1:
        avgWeights= [x/N_FOLDS for x in weight_list]
        x= clust
        pos =[x for x in range(0,len(x))]
        plt.bar(pos,avgWeights, align='center', alpha=0.5)
        plt.xticks(pos, x)
        plt.ylabel('Avg Weight')
        plt.title('Weights of features from Logisitic Regression')
        plt.show()
    return acc_test_a.mean()
        
def feature_select(df,K,dropfeatures,classif):
    """
    Given a dataset (df) it will select a given number k best 
    features to run ML algorithms on
    and return a dataset with K number of features
    """
    # doing the selection without splitting testing and training
    
    df1= df.drop(columns = dropfeatures)
    normalize = MinMaxScaler()
    df_scaled = normalize.fit_transform(df1)
    #fits the data 
    X_fitted = SelectKBest(chi2, k=K).fit(df_scaled,df[classif])
    #getting the transformed array
    X_transformed = X_fitted.fit_transform(df_scaled,df[classif])
    #returns boolean value for which features has been selected
    feature_bool = X_fitted.get_support()
    #gets all the feature names 
    feature_list = df1.columns.tolist()
    
    #getting feature names selected after feature selection
    k_features = []
    for bool,feature in zip(feature_bool,feature_list):
        if bool:
            k_features.append(feature)
    K_DF = pd.DataFrame(X_transformed,columns = k_features)
    K_DF['subject'] = df['subject']
    K_DF['Activity'] = df[classif]
    
    #DfNew = pd.DataFrame(X_new1)
    return K_DF

def run_ML(df,start,stop,filename):
    with open (filename,'w',newline='') as File:
        fieldnames = ['Number', 'Average Accuracy']
        writer = csv.DictWriter(File, fieldnames=fieldnames)
        writer.writeheader()
        #writer.writerow('feature_number,Average Accuracy')        
        for x in range(start,stop-1,5):
            print('running features ',x)
            K_DF = feature_select(df, x,['subject','Activity'],'Activity')
            avg_acc = ml(K_DF,'SVC','subject')
            writer.writerow({'Number': x, 'Average Accuracy': avg_acc})        
            
    print('.......Done running MLmodel.py')

def run_full(df, model_list, subject):
    """
    In this method selected ML models are run
    with all features being used
    """
    for model in model_list:
        print(model)
        acc = ml(df, model, subject)
        
    
    
if __name__ == '__main__':
    
    HARdata = '../data/DataSet_HAR.csv'
    
    HARdf = pd.read_csv(HARdata)
    MLlist = ['SVC','LOGISTIC','KNN']
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-KFeatures', help ='TRUE or FALSE to run feature selection',type = bool,
                        default = False)
    parser.add_argument('-start', help='start number, ex:100', type=int, 
                        default=100)
    parser.add_argument('-stop', help='stopping number', type=int, 
                        default=560)
    parser.add_argument('-o', help='output location', type=str, 
                        default='Accuracy_Features.csv')
    args = parser.parse_args()    
    
    if args.KFeatures:
        run_ML(HARdf,args.start,args.stop,args.o)
    else: 
        run_full(HARdf,['SVC','LOGISTIC'],'subject')
