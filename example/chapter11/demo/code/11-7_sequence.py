#-*- coding: utf-8 -*-
#数据特征分析(画时序图)
import pandas as pd
#参数初始化
inputfile1 = '../data/discdata.xls'
data = pd.read_excel(inputfile1)
data.head()
d = data[(data['ENTITY'] == 'C:\\') & (data['TARGET_ID'] == 184)]
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(9, 7))
import datetime
import matplotlib.dates as mdates
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title(u"C盘已使用空间的时序图")
# ax.set_xlabel(u'日期')
ax.set(xlabel=u'日期', ylabel=u'磁盘使用大小')
# 图上时间间隔显示为10天
ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=range(1, 32), interval=10))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.subplots_adjust(bottom=0.13, top=0.95)
ax.plot(d['COLLECTTIME'], d['VALUE'], 'ro-', color = 'green')
fig.autofmt_xdate()  # 自动根据标签长度进行旋转
'''for label in ax.xaxis.get_ticklabels():   #此语句完成功能同上
       label.set_rotation(45)
'''
plt.savefig('../data/c.jpg')
plt.show()