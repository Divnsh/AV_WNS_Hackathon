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
user_item['count']=user_item[0]
del user_item[0]



