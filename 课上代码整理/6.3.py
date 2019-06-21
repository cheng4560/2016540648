import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = ts.get_hist_data('000001', start='2017-01-01', end='2019-06-03')
#print(df.head(10))

sz = df.sort_index(axis=0, ascending=True)   #排序
sz_return = sz[['p_change']]   #去除数据进行训练
train = sz_return[0:255]  #训练集
test = sz_return[255:]   #测试集
plt.figure(figsize=(10, 5))
train['p_change'].plot()  #训练集可视化
plt.legend(loc='best')
#plt.show()

plt.figure(figsize=(10, 5))
test['p_change'].plot(c='r')  #测试集可视化，改变颜色
plt.legend(loc='best')
plt.show()

train.index = pd.to_datetime(train.index)
test.index = pd.to_datetime(test.index)
dd = np.asarray(train.p_change)  #转换成向量
y_hat = test.copy()   #y_hat和test的长度一样长
y_hat['naive'] = dd[len(dd)-1]    #长度减1分量放在naive中
plt.figure(figsize=(12, 8))
plt.plot(train.index, train['p_change'], label="Train")  #横坐标是使时间，纵坐标是收益率
plt.plot(test.index, test['p_change'], label="Test")
plt.plot(y_hat.index, y_hat['naive'], label="Naive Forcast")
plt.legend(loc='best')
plt.title('Naive Forcast')
#plt.show()

from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(test.p_change, y_hat.naive))  #值越小预测越准
#print(rms)

y_hat_avg = test.copy()
y_hat_avg['avg_forcast'] = train['p_change'].mean()
plt.figure(figsize=(12, 8))
plt.plot(train.index, train['p_change'], label="Train")  #横坐标是使时间，纵坐标是收益率
plt.plot(test.index, test['p_change'], label="Test")
plt.plot(y_hat_avg['avg_forcast'], label='Average Forcast')
plt.legend(loc='best')
#plt.show()
#rms = sqrt(mean_squared_error(test.p_change, y_hat_avg.avg_forcast))
#print(rms)

y_hat_avg=test.copy()
y_hat_avg['moving_avg_forcast'] = train['p_change'].rolling(50).mean().iloc[-1]
plt.figure(figsize=(12,8))
plt.plot(train.index, train['p_change'], label="Train")  #横坐标是使时间，纵坐标是收益率
plt.plot(test.index, test['p_change'], label="Test")
plt.plot(y_hat_avg['moving_avg_forcast'], label="Moving Average Forcast")
plt.legend(loc='best')
plt.show()
