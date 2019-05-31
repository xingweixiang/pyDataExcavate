#-*- coding: utf-8 -*-
#构建并测试逻辑回归模型
import pandas as pd #导入数据分析库

inputfile = '../data/样本数据.xls'
df = pd.read_excel(inputfile, encoding = 'utf-8')

#数据预处理（将销售类型与销售模式以及输出转换成虚拟变量）
type_dummies=pd.get_dummies(df[u'销售类型'],prefix='type')
model_dummies=pd.get_dummies(df[u'销售模式'],prefix='model')
result_dummies=pd.get_dummies(df[u'输出'],prefix='result')
df=pd.concat([df,type_dummies,model_dummies,result_dummies],axis=1)
df.drop([u'销售类型',u'销售模式',u'输出'],axis=1,inplace=True)
#正常列去除，异常列作为结果
df.drop([u'result_正常'],axis=1,inplace=True)
df.rename(columns={u'result_异常':'result'},inplace=True)

#数据划分(80%作为训练数据，20%作为测试数据)
data=df.values
from random import shuffle
shuffle(data)
data_train=data[:int(len(data)*0.8),:]
data_test=data[int(len(data)*0.8):,:]

#确定y值和特征值
y=data_train[:,-1]
x=data_train[:,1:-1]
#逻辑回归
from sklearn import linear_model
clf=linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
#此处的x,y与上文中决策树所用x,y相同
clf.fit(x,y)
#逻辑回归系数
xishu=pd.DataFrame({"columns":list(df.columns)[1:-1], "coef":list(clf.coef_.T)})
from example.chapter16.demo.code.cm_plot import * #导入混淆矩阵可视化函数
#逻辑回归混淆矩阵
cm_plot(y,clf.predict(x)).show()
#对test数据进行预测
predictions=clf.predict(data_test[:,1:-1])
#test混淆矩阵
cm_plot(data_test[:,-1],predictions).show()
