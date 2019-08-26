#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 13:16:01 2019

@author: divyansh

title: Analyis
"""
import pandas as pd
import numpy as np
import scipy
import os
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('/home/vivek/Documents/kaggle_practice/AnalyticsVidya_WNS_clicks/')

train=pd.read_csv(r'./Data/train.csv')
train.head()
    
item=pd.read_csv(r'./Data/item_data.csv')
item.head()

log=pd.read_csv(r'./Data/view_log.csv')
log.head()


# Chi squared test #
from scipy.stats import chi2_contingency,chi2

def chisqtest(baseclass,myclass):
    crosstab=pd.crosstab(train[myclass],train[baseclass])
    chi_res=chi2_contingency(crosstab)
    pvalue = chi_res[1]
    print(pvalue)

def chisqtest_with_dummies(baseclass,myclass):
    crosstab=pd.crosstab(train[myclass],train[baseclass])
    chi_res=chi2_contingency(crosstab)
    pvalue = chi_res[1]
    print(pvalue) 
    dummies = pd.get_dummies(train[myclass])
    for i in range(dummies.shape[1]):
        crosstab=pd.crosstab(dummies.iloc[:,i],train[baseclass])
        chi_res=chi2_contingency(crosstab)
        pvalue=chi_res[1]
        print(pvalue)

chisqtest('is_click','is_4G')    
# pvalue >0.05, therefore having 4G is not significantly related to 'is_click'

chisqtest('is_click','os_version')    
# os version is significantly associated with 'is_active'
chisqtest_with_dummies('is_click','os_version')
# pvalues<0.05/3 => All os versions are significantly associated with 'is_active'

chisqtest('app_code','os_version')    
# Expected values<5 => Result nor reliable

chisqtest('os_version','is_4G')    
# is_4g and os version are significantly associated
dummies = pd.get_dummies(train['os_version'])
for i in range(dummies.shape[1]):
    crosstab=pd.crosstab(dummies.iloc[:,i],train['is_4G'])
    chi_res=chi2_contingency(crosstab)
    pvalue=chi_res[1]
    print(pvalue)


# Plots #

label_counts=train['is_click'].value_counts()
plt.bar(label_counts.index,label_counts.values)
plt.title('Label distribuition')
plt.xticks([0,1])
plt.ylabel('counts')
plt.xlabel('click (Y1 / N0)')
plt.show()
    
def facetgrid(baseclass,myclass):
    g=sns.FacetGrid(train,col=baseclass)
    plt.title(myclass+' vs '+baseclass)
    plt.ylabel('counts')
    plt.xlabel(myclass)
    g.map(plt.hist,myclass)

facetgrid('is_click',"app_code") 

fig,ax=plt.subplots(2,1)  
ax[0].bar(train[train['is_click']==1]['app_code'].value_counts().index,train[train['is_click']==1]['app_code'].value_counts().values)
ax[0].set_title('app code distribution for clicks')
ax[0].set_ylabel('counts')
ax[0].set_xlabel('appcode')

ax[1].bar(train[train['is_click']==0]['app_code'].value_counts().index,train[train['is_click']==0]['app_code'].value_counts().values)
ax[1].set_title('app code distribution for no clicks')
ax[1].set_ylabel('counts')
ax[1].set_xlabel('appcode')
fig.tight_layout()
#fig.suptitle('Appcode distribution across "is_click"',fontsize=20)
plt.show()
# Clicks has more uniform app_code

facetgrid('is_click',"os_version") 

facetgrid('os_version',"app_code") 
# Appcode distribution is similar across os versions

facetgrid('os_version',"is_4G") 

facetgrid('is_4G',"is_click")

 




