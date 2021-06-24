# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:11:39 2019

@author: Akmal
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
#sales dataset
Sales = pd.read_csv('E:\\akmal\\Crop Care Project\\Datasets\\finalsales.csv')
Sales['Date'] = pd.to_datetime(Sales['Date'])
Sales.columns = Sales.columns.str.replace('.','_')

#return dataset
returns = pd.read_csv('E:\\akmal\\Crop Care Project\\Datasets\\Returns.csv')
returns['Date'] = pd.to_datetime(returns.Date)
returns.columns = returns.columns.str.replace(' ','_')
scale = MinMaxScaler()
returns.Products.value_counts()
Sales.Products.value_counts()


def vis_S_R(prod_s,prod_r):
    sales_df = frame(prod_s,Sales)
    ret_df = frame(prod_r,returns)
    ind = sales_df.index
    sales_df['return_qt'] = ret_df
    sales_df =pd.DataFrame(scale.fit_transform(sales_df), index = ind,columns = ('sales_quantity','returned_quantity'))
    sales_df.plot(linewidth=0.5,title = 'Sales and Return of product')
    return sales_df
    
    
def frame(p,data):
    df_1 = data.loc[data.Products == p, ['Date','Quantity']].sort_values(by=['Date'])
    df_1 = df_1.set_index('Date')
    return df_1.resample('M').sum()

#porvide name of product from sales and returns
vis_S_R('KHANJARBIO  -SP 150 GM (1X40)','KHAJANRBIO-SP 150 GM (1X40)')
vis_S_R('XPLODE - 250 ML (1X40)','XPLODE - 250 ML (1X40)')
print(a)
df = returns.loc[returns.Products == 'KHAJANRBIO-SP 150 GM (1X40)', ['Date','Quantity','Dealer_ID']].sort_values(by=['Date'])
df = sales.loc[returns.Products == 'KHAJANRBIO-SP 150 GM (1X40)', ['Date','Quantity','Dealer_ID']].sort_values(by=['Date'])

rfm[rfm.index == 'Del1426'].score
    