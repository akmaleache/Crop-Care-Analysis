# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:43:07 2019

@author: Akmal
"""

from statsmodels.tsa.stattools import adfuller
import numpy as np, pandas as pd
from numpy import log
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


df = pd.read_csv('E:/akmal/Crop Care Project/Datasets/monthly_sales16_18.csv')
result = adfuller(df.Value)
result[0]
result[1]
####  from above p-value  the data is not stationary  ######
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.Value); axes[0, 0].set_title('Original Series')
plot_acf(df.Value, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(df.Value.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df.Value.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(df.Value.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df.Value.diff().diff().dropna(), ax=axes[2, 1])

plt.show()

from pmdarima.arima.utils import ndiffs
y = df.value

## Adf Test
ndiffs(y, test='adf')  # 2

# KPSS test
ndiffs(y, test='kpss')  # 0

# PP test:
ndiffs(y, test='pp')  # 2

dispatch(this.props.actions.guestLoginRequest())

