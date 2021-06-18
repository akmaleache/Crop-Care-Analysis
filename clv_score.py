# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:26:46 2019

@author: Akmal
"""


import pandas as pd
from datetime import date
import numpy as np

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
rmf_i = rfm.sort_values(by = ['total_spend'], ascending = False)
rmf_i = rmf_i.reset_index()

# 20% of top dealers
# 0.20 * 674 = 134.8 = 135
# 80% of total spend
#0.80 * sum(rmf_i['total_spend']) = 332216899.68799925
#sum of top 135 total spend of dealers 
#sum(rmf_i['total_spend'].iloc[1:135])/sum(rmf_i['total_spend'])


from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler()
norm_rfm = pd.DataFrame(norm.fit_transform(rfm))
norm_rfm.index = rfm.index
norm_rfm.columns = ['norm_rec','norm_monetary','norm_freq']
clv = pd.concat([rfm,norm_rfm],axis = 1)

#######      Claculating CLV scores   #############
clv['norm_rec'] = 1-clv['norm_rec']
clv['CLV_score'] =  (0.105*clv['norm_rec'])+(0.258*clv['norm_monetary'])+(0.637*clv['norm_freq'])
clv.to_csv(r'E:/akmal/Crop Care Project/clv_scores.csv')
