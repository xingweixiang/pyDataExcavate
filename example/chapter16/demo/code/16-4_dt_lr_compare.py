#-*- coding: utf-8 -*-
#决策树和逻辑回归模型比较
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
clf=linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6,solver='liblinear')
#此处的x,y与上文中决策树所用x,y相同
clf.fit(x,y)

#决策树模型
from sklearn.tree import DecisionTreeClassifier #导入决策树模型
tree = DecisionTreeClassifier() #建立决策树模型
tree.fit(x, y) #训练

#两个分类方法的ROC曲线
from sklearn.metrics import roc_curve #导入ROC曲线函数
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
fig,ax=plt.subplots()
fpr, tpr, thresholds = roc_curve(data_test[:,-1], tree.predict_proba(data_test[:,1:-1])[:,1], pos_label=1)
fpr2, tpr2, thresholds2 = roc_curve(data_test[:,-1], clf.predict_proba(data_test[:,1:-1])[:,1], pos_label=1)
plt.plot(fpr, tpr, linewidth=2, label = 'ROC of CART', color = 'blue') #作出ROC曲线
plt.plot(fpr2, tpr2, linewidth=2, label = 'ROC of LR', color = 'green') #作出ROC曲线
plt.title('决策树和逻辑回归模型比较')
plt.xlabel('False Positive Rate') #坐标轴标签
plt.ylabel('True Positive Rate') #坐标轴标签
plt.ylim(0,1.05) #边界范围
plt.xlim(0,1.05) #边界范围
plt.legend(loc=4) #图例
plt.show() #显示作图结果
