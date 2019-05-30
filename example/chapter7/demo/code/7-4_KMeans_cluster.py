#-*- coding: utf-8 -*-
#K-Means聚类算法

import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans #导入K均值聚类算法

inputfile = '../tmp/zscoreddata.xls' #待聚类的数据文件
k = 5                       #需要进行的聚类类别数
#读取数据并进行聚类分析
data = pd.read_excel(inputfile) #读取数据
#调用k-means算法，进行聚类分析
kmodel = KMeans(n_clusters = k, n_jobs = 1) #n_jobs是并行数，一般等于CPU数较好
kmodel.fit(data) #训练模型
print(kmodel.cluster_centers_) #查看聚类中心
print(kmodel.labels_) #查看各样本对应的类别
labels = kmodel.labels_#查看各样本类别
demo= DataFrame(kmodel.cluster_centers_, columns=data.columns) # 保存聚类中心
# demo2= demo['numbers'].value_counts() # 确定各个类的数目
#画雷达图 客户群特征分析图
data = demo.values
from example.chapter7.demo.code.radar import drawRader
title = 'RadarPicture'
rgrids = [0.5, 1, 1.5, 2, 2.5]
itemnames = ['ZL','ZR','ZF','ZM','ZC']
labels = ['重要保持客户','重要发展客户','重要挽留客户','一般客户','低价值客户']
drawRader(itemnames=itemnames,data=data,title=title,labels=labels, saveas = '2.jpg',rgrids=rgrids)