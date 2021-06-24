# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:43:31 2019

@author: Akmal
"""
import re
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
Sales = pd.read_csv('E:\\akmal\\Crop Care Project\\Datasets\\finalsales.csv')
#Sales = pd.read_csv('E:\\akmal\\Crop Care Project\\Datasets\\Sales.csv')

Sales['Date'] = pd.to_datetime(Sales['Date'])
Sales.isnull().sum()#voucher_ref = 7670, Value = 13 missing
Sales = Sales.dropna()
Sales.columns = Sales.columns.str.replace('.','_')
#Sales.columns = Sales.columns.str.replace(' ','_')
Sales.columns

Sales.dtypes
Sales.describe()
a = Sales.describe(include = 'object')
Sales.Dealer_ID.nunique()
Sales.Dealer_ID.unique()
Sales.Dealer_ID.value_counts()
Sales.Date.describe()
Sales.Products.nunique()
Sales.Area.nunique()
Sales.Products.value_counts()

len(Sales[Sales.Value<5].Dealer_ID)
products_index = pd.Index(Sales.Products)
products_index.value_counts()[0]

#missing value or irrelevent value
len(Sales.loc[Sales.Value == 1,['Value','Area','Products','Dealer_ID']])
len(Sales.loc[Sales.Value == Sales.Quantity,['Value','Area','Products','Dealer_ID']])

#for i in range(0,178):
#   if products_index.value_counts()[i]>50:
#       print(products_index.values[0])

len(Sales['Products'].value_counts()>50)
Sales['Products'].unique()

plt.figure()
df = Sales.loc[Sales.Products == 'XPLODE - 250 ML (1X40)', ['Date','Quantity']].sort_values(by=['Date'])
df = df.set_index('Date')
dfm = df.resample('M').sum()
dfm['Quantity'].plot(linewidth=0.5)

plt.figure()
df_1 = Sales.loc[Sales.Products == 'KHANJARBIO  -SP 150 GM (1X40)', ['Date','Quantity']].sort_values(by=['Date'])
df_1 = df_1.set_index('Date')
dfm_1 = df_1.resample('M').sum()
dfm_1['Value'].plot(linewidth=0.5)

#checking Value is equal for each product for deffirent dealers
sales_2 = Sales.copy()
sales_2['Quantity'] = sales_2['Quantity'].astype(float)
#removing products having value = 1
sales_2.drop(sales_2[(sales_2.Value == 1)| (sales_2.Value == sales_2.Quantity)].Value.index,inplace = True,axis = 0)
sales_2.loc[sales_2.Value == 1,['Area','Products']]

#cal_value = sales_2.Quantity * unit_value
sales_2['unit_value'] = sales_2.Value/sales_2.Quantity
sales_2.loc[sales_2.Products == 'XPLODE - 250 ML (1X40)',['unit_value','Dealer_ID','Area','Date']].sort_values(by = ['Date'])
sales_2.loc[sales_2.Products == 'KHANJARBIOSL- 250 ML (1X40)',['unit_value','Dealer_ID','Area']]
sales_2.loc[sales_2.Products == 'KHANJARBIO  SL- 500 ML (1X20)',['unit_value']]

#Sales['unit_value']= Sales.Value / Sales.Quantity
##impute Value of product with its unit price 
#Sales.groupby('Products').unit_value.min()
#Sales.loc[Sales.Products == ' TIGER 708 POWER SPRAYER-20 LTR ',['Dealer_ID','unit_value','Date','Quantity','Area']]
#np.nanmedian(Sales[Sales.Products == 'XPLODE - 250 ML (1X40)'].unit_value)
#sales_2.loc[sales_2.Products == 'XPLODE - 250 ML (1X40)',['Dealer_ID','unit_value','Date','Quantity','Area']]

#
sales_2['nanmedian_price'] = sales_2.Products.apply(lambda x: np.nanmedian(sales_2[sales_2.Products == x].unit_value))
sales_2.isnull().sum()

#combining two tables to visualize the sales and return (execute below code and then go to return.py line number 28)
#plt.figure()
ts_sakti = Sales.loc[Sales.Products == 'SHAKTI - 1 LTR (1X10)', ['Date','Quantity']].sort_values(by=['Date'])
ts_sakti = ts_sakti.set_index('Date')
dfm_sakti = ts_sakti.resample('M').sum()

dfm_sakti['Quantity'].plot(linewidth=0.5)

ts_xplode = Sales.loc[Sales.Products == 'XPLODE - 250 ML (1X40)', ['Date','Quantity']].sort_values(by=['Date'])
ts_xplode = ts_xplode.set_index('Date')
dfm_xplode = ts_xplode.resample('M').sum()



##########  adding farming seasons to data #############

def decide_season(cdate):
    if(cdate >=  datetime.datetime(2017,7,1) and cdate <= datetime.datetime(2017,10,25)):
        return str('kharif')
    if(cdate >  datetime.datetime(2017,10,25) and cdate <= datetime.datetime(2018,3,31)):
        return str('rabi')
    if(cdate >=  datetime.datetime(2017,4,1) and cdate <= datetime.datetime(2017,7,1)):
        return str('zaid')

season = []
Sales.Date.describe()
Sales['season'] = Sales.Date.apply(lambda x: decide_season(x))
Sales[Sales.season.isnull()].Date
Sales[Sales.Products == 'SUPERSTAR - 20 KG (1X1)'].season.value_counts()
Sales.pivot_table('Value','season')
Sales.season.value_counts()
Sales.


#decom = PCA(n_components = 2)
#a = decom.fit_transform(norm_rfm)
#sum(decom.explained_variance_ratio_)
#pca_data = pd.DataFrame(data = a,columns = ['pc1','pc2'])
#fig = plt.figure()
#plt.scatter(pca_data['pc1'],pca_data['pc2'])


from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler()
norm_rfm = pd.DataFrame(norm.fit_transform(rfm))
norm_rfm.index = rfm.index
norm_rfm.columns = ['norm_rec','norm_monetary','norm_freq']
rfm = pd.concat([rfm,norm_rfm],axis = 1)

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 
kmean = KMeans(n_clusters = 3)
model = kmean.fit(rfm.iloc[:,3:])
TWSS = [] # variable for storing total within sum of squares for each kmeans
k = list(range(2,10))
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(rfm.iloc[:,3:])
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(rfm.iloc[:,3:].iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,rfm.iloc[:,3:].shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

plt.plot(k,TWSS,'ro-')

kmeans = KMeans(n_clusters = 5)
rfm['score'] = kmeans.fit_predict(rfm.iloc[:,3:])
kmeans.labels_

#clustring 
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax = plt.axes(projection = '3d')
ax.scatter(rfm['norm_rec'],rfm['norm_freq'],rfm['norm_tot'],c = rfm['score'],edgecolor = 'none',label='score')
ax.set_xlabel('Recency');ax.set_ylabel('Frequency');ax.set_zlabel('Monetary')
ax.legend(bbox_to_anchor = (1.5, 1))
plt.show()
colors = ['r', 'b', 'y', 'c']
for i, c in enumerate(clusters):
    ax.scatter(c[0], c[1], c[2], c=colors[i], label='cluster {}'.format(i))

a = rfm.groupby('score').agg('recent_days')
des = rfm.describe()

rfm.pivot_table(['recent_days','total_spend','frequent_days'],'score')
rfm.pivot_table(['recent_days','total_spend','frequent_days'],'score',aggfunc = np.min)
rfm.pivot_table(['recent_days','total_spend','frequent_days'],'score',aggfunc = np.max)
#rfm['RFM_Score'] = rfm.q_recent.astype(str) + rfm.q_frequency.astype(str) + rfm.q_monetary.astype(str)
#rfm.RFM_Score = rfm.RFM_Score.astype(int)
#
#rfm[rfm['RFM_Score']==111].sort_values('q_monetary',ascending = False).head()
#rfm.sort_values(by = 'RFM_Score')

################ Calculating CLV(Customer Lifetime Value) #######################

clv = Sales.groupby('Dealer_ID').agg({'Date': lambda d:(Sales.Date.max() - d.max()).days ,
                                        'Dealer_ID':lambda d: len(d),
                                        'Quantity':lambda q: q.sum(),
                                        'Value':lambda price: price.sum()})
clv.columns = ['recent_days','transaction_freq','tot_quantity','tot_price']

clv['avg_order_value'] = clv.tot_price/clv.transaction_freq 
purchase_freq = sum(clv.transaction_freq)/clv.shape[0]
repeat_rate = clv[clv.transaction_freq > 1].shape[0]/clv.shape[0]
churn_rate = 1-repeat_rate

#profit margin
clv['profit_margin'] = clv.tot_price*0.5
#customer value
clv['customer_val'] = (clv.avg_order_value * purchase_freq)/churn_rate

#clv
clv['CLV'] = clv.customer_val*clv.profit_margin


