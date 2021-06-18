# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:11:59 2019

@author: Akmal
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

returns = pd.read_csv('E:\\akmal\\Crop Care Project\\Datasets\\Returns.csv') 
returns.isnull().sum()
returns.dtypes
#preprocesssing
returns['Date'] = pd.to_datetime(returns.Date)
returns.columns = returns.columns.str.replace(' ','_')
returns.Date.describe()
#total no of dealers who have returned
returns['Dealer_ID'].nunique()
returns['Dealer_ID'].value_counts()

#no. of products and count
returns['Products'].nunique()
returns['Products'].value_counts()

#sorting top products returns by date
returns.loc[returns.Products == 'MADALI-100GRX40', ['Date', 'Dealer_ID','Quantity']].sort_values(by = 'Date')
returns.loc[returns.Products == 'SHAKTI- 1 LTR (1X10)', ['Date', 'Dealer_ID','Quantity']].sort_values(by = 'Date')
help(pd.resample)
#plot for sum of quantity(Continued from sales.py from line no 78)

ts_sakti_r = returns.loc[returns.Products == 'SHAKTI- 1 LTR (1X10)', ['Date','Quantity']].sort_values(by=['Date'])
ts_sakti_r = ts_sakti_r.set_index('Date')
dfm_sakti_r = ts_sakti_r.resample('M').sum()
dfm_sakti_r['Sales_Q'] = dfm_sakti['Quantity'] 
normalized_df=(dfm_sakti_r-dfm_sakti_r.min())/(dfm_sakti_r.max()-dfm_sakti_r.min())
normalized_df.plot(linewidth=0.5,title = 'Sales and Return of product sakti')


ts_xplode_r = returns.loc[returns.Products == 'XPLODE - 250 ML (1X40)', ['Date','Quantity']].sort_values(by=['Date'])
ts_xplode_r = ts_xplode_r.set_index('Date')
dfm_xplode_r = ts_xplode_r.resample('M').sum()
dfm_xplode_r['Sales_Q'] = dfm_xplode['Quantity'] 
normalized_df=(dfm_xplode_r-dfm_xplode_r.min())/(dfm_xplode_r.max()-dfm_xplode_r.min())
normalized_df.plot(linewidth=0.5,title = 'Sales and Return of product xplode')


ts_sakti_r = returns.loc[returns.Products == 'Del121', ['Date','Quantity']].sort_values(by=['Date'])
ts_sakti_r = ts_sakti_r.set_index('Date')
dfm_sakti_r = ts_sakti_r.resample('M').sum()