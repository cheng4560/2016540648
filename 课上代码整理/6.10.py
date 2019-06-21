import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams#可设图形大小
rcParams['figure.figsize'] = 15, 6

#读取数据,查看数据
data = pd.read_csv('AirPassengers.csv')
print(data.head())
print('\n data Types:')
print(data.dtypes)

#转换数据类型
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m') #Lambda,格式固定。
#重新读取数据，读取时告诉用dateparse转换日期
data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month', date_parser=dateparse)
print(data.head())
data.index

ts = data['#Passengers']
print(ts.head(10))
print(ts['1949'])

#
plt.plot(ts)
plt.show()

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
   rolmean = pd.Series(timeseries).rolling(window=12).mean()
   rolstd = pd.Series(timeseries).rolling(window=12).std()
   orig = plt.plot(timeseries, color='blue', label='Orignal')
   mean = plt.plot(rolmean, color='red', label='Rolling Mean')
   std = plt.plot(rolstd, color='black', label='Rolling Std')
   plt.legend(loc='best')
   plt.title('Rolling Mean & Standard Deviation')
   plt.show()
   print('Results of Dickey_Fuller Test:')
   dftest = adfuller(timeseries, autolag='AIC')
   dfoutput = pd.Series(dftest[0:4], index=['Test Statisic', 'p-value', '#LagsUsed', 'Number of Obersercation Used'])
   for key, value in dftest[4].items():
       dfoutput['Critical Value(%s)' % key] = value
   print(dfoutput)
test_stationarity(ts)
#mean、std必须时两者都稳定，才认为数据是稳定的