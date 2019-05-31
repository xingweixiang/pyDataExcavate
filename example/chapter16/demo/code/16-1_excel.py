#-*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt #导入Matplotlib
inputfile = '../data/样本数据.xls'
df = pd.read_excel(inputfile, encoding = 'utf-8')

fig=plt.figure()
fig.set(alpha=0.2)
#不同销售类型和销售模式下的偷漏税情况
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
plt.subplot2grid((1,2),(0,0))
df_type=df[u'销售类型'][df[u'输出']=='异常'].value_counts()
df_type.plot(kind='bar',color='Red')
plt.title(u'不同销售类型下的偷漏税情况',fontproperties='SimHei')
plt.xlabel(u'销售类型',fontproperties='SimHei')
plt.ylabel(u'异常数',fontproperties='SimHei')
plt.subplot2grid((1,2),(0,1))
df_model=df[u'销售模式'][df[u'输出']=='异常'].value_counts()
df_model.plot(kind='bar',color='Orange')
plt.title(u'不同销售模式下的偷漏税情况',fontproperties='SimHei')
plt.xlabel(u'销售模式',fontproperties='SimHei')
plt.ylabel(u'异常数',fontproperties='SimHei')
plt.subplots_adjust(wspace=0.3)
plt.show()

#不同输出情况下的数值型变量总体情况
df_normal=df.iloc[:,3:][df[u'输出']=='正常'].describe().T
df_normal=df_normal[['count','mean','max','min','std']]
df_abnormal=df.iloc[:,3:][df[u'输出']=='异常'].describe().T
df_abnormal=df_abnormal[['count','mean','max','min','std']]
print(df_normal)
print(df_abnormal)