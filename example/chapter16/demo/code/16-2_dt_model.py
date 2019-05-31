#-*- coding: utf-8 -*-
#构建并测试CART决策树模型
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
from sklearn.tree import DecisionTreeClassifier #导入决策树模型
treefile = '../tmp/tree.pkl' #模型输出名字
tree = DecisionTreeClassifier() #建立决策树模型
tree.fit(x, y) #训练

#保存模型
from sklearn.externals import joblib
joblib.dump(tree, treefile)

from example.chapter16.demo.code.cm_plot import * #导入混淆矩阵可视化函数
cm_plot(y, tree.predict(x)).show() #显示混淆矩阵可视化结果

cm_plot(data_test[:,-1],tree.predict(data_test[:,1:-1])).show()