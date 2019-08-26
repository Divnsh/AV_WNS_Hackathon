#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 11:32:47 2019

@author: divyansh

title: AV Hackathon WNS
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

os.chdir('/home/vivek/Documents/kaggle_practice/AnalyticsVidya_WNS_clicks/')
color = sns.color_palette()
score_fn=make_scorer(f1_score,average='weighted')

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

if __name__=='__main__':
    train=pd.read_csv(r'./Data/train.csv')
    item=pd.read_csv(r'./Data/item_data.csv')
    item.head()
    log=pd.read_csv(r'./Data/view_log.csv')
    log.head()
    train.head()
    log = log.merge(item,on='item_id',how='inner')
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
    
from datetime import timedelta,datetime as dt
    
v63410=train[train['user_id']==63410].sort_values('impression_time')
l63410=log[log['user_id']==63410].sort_values('server_time')   


def check_dates(row):
    j=0
    for i in pd.to_datetime(sorted(log[log['user_id']==63410]['server_time'])):
        j=j+(pd.to_datetime(row['impression_time'])>i and pd.to_datetime(row['impression_time'])<i+timedelta(days=7)) 
    return j     

res=train[train['user_id']==63410].apply(lambda x:check_dates(x),axis=1)

sorted(log[log['user_id']==63410]['server_time'])

sorted(log[log['user_id']==63410]['server_time']).apply(lambda x:>=x, <x+timedelta(days=7))
























