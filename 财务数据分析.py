import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tushare as ts

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

sh=ts.get_k_data(code='sh', ktype='D', autype='qfq', start='1990-12-20')
print(sh.head(5))
sh.index = pd.to_datetime(sh.date)
sh['close'].plot(figsize=(12, 6))
plt.title('Trend chart for SH stocks from 1990 to now')
plt.xlabel('date')
plt.show()

print(sh.describe().round(2))
print(sh.count())
print(sh.close.mean())
print(sh.open.std())

sh.loc["2007-01-01":]['volume'].plot(figsize=(12,6))
plt.title('from 01/01/2016')
plt.show()

ma_day=[20, 52, 252]
for ma in ma_day:
    column_name = "%sday mean"%(str(ma))
    sh[column_name] = sh['close'].rolling(ma).mean()
sh.tail(3)

sh.loc['2007-01-01':][["close", "20day mean", "52day mean", "252day mean"]].plot(figsize=(12, 6))
plt.title('2007 to now the trend of CHN Stock Market')
plt.xlabel('Date')
plt.show()

sh["Daily profit"] = sh["close"].pct_change()
sh["Daily profit"].loc['2016-01-01':].plot(figsize=(12, 6))
plt.xlabel('Date')
plt.ylabel('Daily profit')
plt.title('From 2006 to now daily profit')
plt.show()

sh["Daily profit"].loc['2006-01-01':].plot(figsize=(12, 4), linestyle="--", color="g")
plt.xlabel('Date')
plt.show()


stocks={'上证指数': 'sh', '深证指数': 'sz', '沪深300': 'hs300', '上证50': 'sz50', '中小指数': 'zxb', '创业板': 'cyb'}
stock_index = pd.DataFrame()
for stock in stocks.values():
    stock_index[stock] = ts.get_k_data(stock, ktype='D', autype='qfq', start='2005-01-01')['close']
print(stock_index.head())

def return_risk(stocks,stsrtdate='2016-1-1'):        #计算风险率
    close = pd.DataFrame()
    for stock in stocks.value():
        close[stock] = ts.get_k_data(stock, ktype='D', autype='qfq', start=startdate)['close']
        tech_rets = close.pct_change()[1:]    #算收益
        rets = tech_rets.dropna()     #清洗数据，除去空值
        ret_mean = rets.mean()*100
        ret_std = rets.std()*100
        return ret_mean, ret_std

def plot_return_risk():        #风险可视化
    ret, vol=return_risk(stocks)
    color = np.array([0.18, 0.96, 0.75, 0.3, 0.9, 0.5])
    plt.scatter(ret, vol, marker='o', c=color, s=500, camp=plt.get_cmap('Spectral'))
    plt.xlabel("日收益率均值%")
    plt.ylabel("标准差%")
    for label, x, y in zip(stocks.keys(), ret, vol):
        plt.annotate(label, xy=(x, y), xytext=(20, 20), textcoords="offset points", ha="right", va="bottom", bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5), arrowprops=dict(arrowstyle="->", connetionstyle="arc3,rad=0"))
    stocks = {'上证指数': 'sh', '深证指数': 'sz', '沪深300': 'hs300', '上证50': 'sz50', '中小指数': 'zxb', '创业板': 'cyb'}
    plot_return_risk()

    stocks = {'中国平安': '601318', '格力电器': '000651', '徐工机械': '000425', '招商银行': '600036', '恒生电子': '600570', '贵州茅台': '600519'}
    startdate = '2018-1-1'
    plot_return_risk()

df=ts.get_k_data('sh', ktype='D', autype='qfq', start='2006-1-1')
df.index = pd.to_datetime(df.date)
tech_rets = df.close.pct_change()[1:]
rets = tech_rets.dropna()
print(rets.head(100))