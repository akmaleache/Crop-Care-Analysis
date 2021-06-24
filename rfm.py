# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 19:33:25 2019

@author: Akmal
"""

import re
import pandas as pd
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
Sales = pd.read_csv('E:\\akmal\\Crop Care Project\\Datasets\\finalsales.csv')
Sales = Sales.drop('Unnamed: 0',axis = 1)
Sales['Date'] = pd.to_datetime(Sales['Date'])
Sales.isnull().sum()#voucher_ref = 7670, Value = 13 missing
Sales.columns = Sales.columns.str.replace('.','_')

#########calculating RFM (Recency, frequency, monetery) for customer worthiness


rfm = Sales.groupby('Dealer_ID').agg({'Date': lambda d: (Sales.Date.max() - d.max()).days,
                                            'Value': lambda price: price.sum()})
rfm['frequent_days'] = Sales.groupby('Dealer_ID').agg({'Date': lambda d: d.nunique()})
rfm.columns = ['recent_days','total_spend','frequent_days']

#rfm.corr()
#rfm.dtypes
#Sales[Sales['Dealer_ID'] == 'Del1114']
rfm.dtypes
##  RFM values
rfm['q_recent'] = pd.qcut(rfm.recent_days,5,[5,4,3,2,1],duplicates='drop')
rfm['q_frequency'] = pd.qcut(rfm.frequent_days,6,[1,2,3,4,5],duplicates='drop')
rfm['q_monetary'] = pd.qcut(rfm.total_spend,5,[1,2,3,4,5])
rfm['RFM_Score'] = rfm.q_recent.astype(str)+ rfm.q_frequency.astype(str) + rfm.q_monetary.astype(str)


quintiles = rfm[['recent_days', 'total_spend', 'frequent_days']].quantile([.2, .4, .6, .8]).to_dict() 

#segmentation
segt_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at risk',
    r'[1-2]5': 'can\'t loose',
    r'3[1-2]': 'about to sleep',
    r'33': 'need attention',
    r'[3-4][4-5]': 'loyal customers',
    r'41': 'promising',
    r'51': 'new customers',
    r'[4-5][2-3]': 'potential loyalists',
    r'5[4-5]': 'champions'
}
 
rfm['Segment'] = rfm['q_recent'].astype(str) + rfm['q_frequency'].astype(str)
rfm['Segment'] = rfm['Segment'].replace(segt_map, regex=True)




