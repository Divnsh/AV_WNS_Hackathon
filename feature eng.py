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
from datetime import timedelta,datetime as dt
import multiprocessing
from itertools import repeat
from functools import reduce
import gc
import time
from pathos.multiprocessing import ThreadingPool
import traceback


os.chdir('/home/divyansh/DataScience/Modelling/AV_WNS_Hackathon')
color = sns.color_palette()
score_fn=make_scorer(f1_score,average='weighted')
          

def load_data():
    train=pd.read_csv(r'./Data/train.csv')
    train['impression_time']=pd.to_datetime(train['impression_time'])
    item=pd.read_csv(r'./Data/item_data.csv')
    item.head()
    log=pd.read_csv(r'./Data/view_log.csv')
    log['server_time']=pd.to_datetime(log['server_time'])
    log.head()
    train.head()
    log = log.merge(item,on='item_id',how='inner')
    return train,log
    
train,log=load_data()

# Users split function
#train=train[:10000]   
users=train.user_id.unique()

def users_split(k):
    u=[]
    for i in range(1,k+1):
        u.append(users[(i-1)*int(len(users)/k):i*int(len(users)/k)])
    if len(users)%k!=0:
        u.append(users[i*int(len(users)/k):])
    return u

userslist = users_split(3) 

# Checking for ads shown within 7 days

def check_dates(row,user_id):
    logdf = log.loc[log['user_id']==user_id].sort_values('server_time')
    if len(logdf)==0:
        return [[] for i in range(len(log.columns))]
    is_ad = logdf.apply(lambda x:(row['impression_time']>=x['server_time']) & (row['impression_time']<=x['server_time']+timedelta(days=7)),axis=1)
    #print(is_ad)
    colnames = list(log.columns)
    colnames.remove('server_time')
    colnames.remove('user_id')
    finlist = [list(is_ad.astype('int'))]
    for col in colnames:
        finlist.append(list(logdf[is_ad][col]))
    #print(finlist)    
    return finlist   

# Tracking ads shown within 7 days
print_lock= multiprocessing.Lock()    
def track(u):
    try:           
        t=train.loc[train.user_id==u].copy()
        t['is_tracked']= t.apply(lambda x:check_dates(x,u),axis=1)
        return t
    except:
         with print_lock:
            print("%s: An exception occurred" % u)
            print(traceback.format_exc())
    
def add_tracking(userslist):
    print(userslist)
    fdf = []
        
    with multiprocessing.Pool(processes=8) as pool:
        fdf = pool.map(track, userslist)
        pool.close()
        pool.join()
    return reduce(lambda x,y: pd.concat([x,y],axis=0),fdf)    

start = time.time()

result=ThreadingPool().map(add_tracking, list(userslist))

'''
with multiprocessing.Pool(processes=8) as pool:
    result = pool.map(add_tracking, list(userslist))
'''
        
'''
fdf=[]
for u in userslist:
    res=add_tracking(u)
    fdf.append(res)
'''  
end = time.time()       
minutes=int((end-start)/60)
seconds=np.round((end-start)%60,2)
print("time taken: {} minutes and {} seconds".format(minutes,seconds))


fdf = reduce(lambda x,y: pd.concat([x,y],axis=0),result)

fdf=fdf.reset_index(drop=True)

cols=list(log.columns)
cols.remove('user_id')
cols[0] = 'n_trackings'

for i,col in enumerate(cols):
    fdf[col]=fdf.is_tracked.apply(lambda x:x[i])
    
fdf.drop(['is_tracked'],axis=1).to_csv('/home/divyansh/DataScience/Modelling/AV_WNS_Hackathon/Data/train_new.csv',index=False,header=True)    
    
    











