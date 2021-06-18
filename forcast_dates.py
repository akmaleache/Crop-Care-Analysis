# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:55:10 2019

@author: Akmal
"""

import pandas as pd

sales16_17 = pd.read_excel('E:/akmal/Crop Care Project/Datasets/Sales(16-17).xlsx')
sales17_18 = pd.read_csv('E:/akmal/Crop Care Project/Datasets/Sales(17-18).csv')
sales16_17.isnull().sum()
sales17_18.isnull().sum()
sales16_17 = sales16_17.drop(sales16_17[sales16_17['Quantity']==sales16_17['Value']].index)
sales17_18 = sales17_18.drop(sales17_18[sales17_18['Quantity']==sales17_18['Value']].index)



#sales 16-17 dates are already in datatime format, changing 17-18 date format before concating
sales17_18['Date'] = pd.to_datetime(sales17_18['Date'])
sales16_17 = sales16_17.drop([12722])
sales17_18 = sales17_18.drop([9335])

sales = pd.concat([sales16_17,sales17_18])
sales.to_csv(r'E:/akmal/Crop Care Project/Datasets/sales16_18.csv')

monthly_sales16_18 = sales.resample('M', on = 'Date').sum()
weekly_sales16_18 = sales.resample('W', on = 'Date').sum()

monthly_sales16_18.to_csv(r'E:/akmal/Crop Care Project/Datasets/monthly_sales16_18.csv')
weekly_sales16_18.to_csv(r'E:/akmal/Crop Care Project/Datasets/weekly_sales16_18.csv')
