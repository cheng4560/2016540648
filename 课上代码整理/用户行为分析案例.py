#导入常用的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
plt.style.use('ggplot')  #更改设计风格，使用自带的形式进行美化，这是一个r语言的风格
#导入源数据
columns = ['用户ID', '购买日期', '订单数', '订单金额']
df = pd.read_csv("CDNOW_master.txt", names=columns, sep='\s+')
print(df.head(10))
print(df.describe())
print(df.info())
# 将购买日期列进行数据类型转换
df['购买日期'] = pd.to_datetime(df.购买日期, format='%Y%m%d') #Y四位数的日期部分，y表示两位数的日期部分
df['月份'] = df.购买日期.values.astype('datetime64[M]')
print(df.head())
print(df.info())
# 解决中文显示参数设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 设置图的大小，添加子图
plt.figure(figsize=(15, 12))
# 每月的总销售额
plt.subplot(221)
df.groupby('月份')['订单金额'].sum().plot(fontsize=24)
plt.title('总销售额', fontsize=24)

# 每月的消费次数
plt.subplot(222)
df.groupby('月份')['购买日期'].count().plot(fontsize=24)
plt.title('消费次数', fontsize=24)

# 每月的销量
plt.subplot(223)
df.groupby('月份')['订单数'].sum().plot(fontsize=24)
plt.title('总销量', fontsize=24)

# 每月的消费人数
plt.subplot(224)
df.groupby('月份')['用户ID'].apply(lambda x: len(x.unique())).plot(fontsize=24)
plt.title('消费人数', fontsize=24)
plt.tight_layout()  # 设置子图的间距
plt.show()
# 根据用户id进行分组
group_user = df.groupby('用户ID').sum()
group_user.describe()
#查询条件：订单金额 < 4000
group_user.query('订单金额 < 4000').plot.scatter(x='订单金额', y='订单数')
group_user.订单金额. plot.hist(bins=20)
#bins = 20,就是分成20块，最高金额是14000，每个项就是700
group_user.query("订单金额< 800")['订单金额'].plot.hist(bins=20)
#每个用户的每次购买时间间隔
order_diff = df.groupby('用户ID').apply(lambda x: x['购买日期'] - x['购买日期'].shift())
order_diff.head(10)
order_diff.describe()
plt.figure(figsize=(15, 5))
plt.hist((order_diff / np.timedelta64(1, 'D')).dropna(), bins=50)
plt.xlabel('消费周期', fontsize=24)
plt.ylabel('频数', fontsize=24)
plt.title('用户消费周期分布图', fontsize=24);
orderdt_min = df.groupby('用户ID').购买日期.min()#第一次消费
orderdt_max = df.groupby('用户ID').购买日期.max()#最后一次消费
(orderdt_max-orderdt_min).head()
(orderdt_max-orderdt_min).mean()
((orderdt_max-orderdt_min)/np.timedelta64(1, 'D')).hist(bins=15)
'''因为数据类型是timedelta时间，无法直接作出直方图，所以先换算成数值。换算的方式直接除timedelta函数即可，
np.timedelta64(1, ‘D’)，D表示天，1表示1天，作为单位使用的。因为max-min已经表示为天了，两者相除就是周期'''
#计算所有消费过两次以上的老客的生命周期
life_time = (orderdt_max - orderdt_min).reset_index()
life_time.head()
#用户生命周期分布图
plt.figure(figsize=(10, 5))
life_time['life_time'] = life_time.购买日期 / np.timedelta64(1, 'D')
life_time[life_time.life_time > 0].life_time.hist(bins=100, figsize=(12, 6))
plt.show()
#去掉0天生命周期的用户之后的用户生命周期的平均值
life_time[life_time.life_time > 0].购买日期.mean()
rfm = df.pivot_table(index='用户ID',
                     values=['订单金额', '购买日期', '订单数'],
                     aggfunc={'订单金额': 'sum',
                              '购买日期': 'max',
                              '订单数': 'sum'})
print(rfm.head())
