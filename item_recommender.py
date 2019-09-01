# -*- coding: utf-8 -*-
"""
User-Item preference - recommendation
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

def load_data():
    train=pd.read_csv(r'./Data/train_new.csv')
    train['impression_time']=pd.to_datetime(train['impression_time'])
    item=pd.read_csv(r'./Data/item_data.csv')
    log=pd.read_csv(r'./Data/view_log.csv')
    log['server_time']=pd.to_datetime(log['server_time'])
    log = log.merge(item,on='item_id',how='inner')
    return train,log

train,log = load_data()

cols=list(log.columns)
cols.remove('server_time')
train[cols]

# User item implicit data
user_item=log.groupby(['user_id','item_id']).size().reset_index()
user_item['freq']=user_item[0]
del user_item[0]

# Implicit ALS
import implicit
import scipy
from implicit.evaluation import train_test_split,mean_average_precision_at_k,ndcg_at_k

# sparse matrix
user_item.set_index(['user_id', 'item_id'], inplace=True)
user_items = scipy.sparse.coo_matrix((user_item.freq, (user_item.index.labels[0], user_item.index.labels[1])))
#user_items=scipy.sparse.csr_matrix(user_item.values.T)

# initialize a model
model = implicit.als.AlternatingLeastSquares(factors=50)

# train the model on a sparse matrix of item/user/confidence weights
def train_als(trn,tst):
    model.fit(trn,show_progress=True)  
    MAP = mean_average_precision_at_k(model, trn, tst, K=10,
                                      show_progress=True, num_threads=1)
    NDCG = ndcg_at_k(model, trn, tst, K=10,
                  show_progress=True, num_threads=1)
    print("MAP is %.4f and NDCG is %.4f: " %(MAP,NDCG))
    return model


# train test split validation
trn,tst = train_test_split(user_items, train_percentage=0.8)
model = train_als(trn,tst)

# Fitting model on the whole data
model = train_als(user_items,user_items)

# recommend items for a user
user_items1=user_items.T.tocsr()
recommendations = model.recommend(72172, user_items1,recalculate_user=True)

# find related items
related = model.similar_items(64408)

