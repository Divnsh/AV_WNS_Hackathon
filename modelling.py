# -*- coding: utf-8 -*-
"""
title: Modelling on final data
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,make_scorer
from sklearn.decomposition import PCA
from yellowbrick.classifier import ROCAUC ,ClassificationReport,ConfusionMatrix
import os
from datetime import timedelta,datetime as dt
import multiprocessing
from itertools import repeat
from functools import reduce
import gc
import time
from pathos.multiprocessing import ThreadingPool
import traceback

def dimesionality_reduction(vect):
    pca=PCA(n_components=2)
    d2_array=pca.fit_transform(vect)
    return pca,d2_array

def plot_reduced_dimension(d2_array,label):
    plt.scatter(d2_array[:,0],d2_array[:,1],c=label,cmap='viridis')
    plt.colorbar()
    plt.title('2D Plot')
    plt.show()

def cross_validation(model,x,y,cv=3):
    cv_res=cross_validate(model,x,y,return_train_score=True,scoring=score_fn,cv=cv)
    return cv_res

def plot_cv_res(cv_res):
    plt.plot(cv_res['test_score'])
    plt.title('Test score')
    plt.show()
  
    plt.plot(cv_res['train_score'])
    plt.title('Train score')
    plt.show()
    return

def baseline_model(df):
    X_train=df.drop('is_click',axis=1)
    with open(r'./baseline_logistic.pkl','wb') as f:
        pickle.dump(X_train,f)
    logistic=LogisticRegression()
    cv_res=cross_validation(logistic,X_train,df['is_click'])
    plot_cv_res(cv_res)
    return logistic.fit(X_train,df['is_click']),X_train


def predict_test_values(test_df,model,transformer):
    vect=transformer.transform(test_df['tweet'])
    predictions=model.predict(vect)
    df=pd.DataFrame()
    df['id']=test_df['id'].values
    df['label']=predictions
    return  df

log,X_train=baseline_model(train)    
test_df=pd.read_csv(r'./Data/test.csv')
with open(r'./Models/baseline_logistic.pkl','rb') as f:
        X_train=pickle.load(f)
pred_df=predict_test_values(test_df,log,X_train)
pred_df.to_csv('./Submissions/baseline_predictions.csv',index=False)
#pca,arr=dimesionality_reduction(X_train.toarray())
#plot_reduced_dimension(arr,train['is_click'])
visualizer = ROCAUC(log)
visualizer.score(X_train,train['label'])
visualizer.poof()
feature_series=pd.Series(index=X_train.columns,data=X_train.values)