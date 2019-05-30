#-*- coding: utf-8 -*-
# 采用ROC曲线评价方法来测试评估模型分类的性能，一个优秀的分类器应该是尽量靠近左上角的
import pandas as pd
from random import shuffle#导入随机函数shuffle，用来打乱数据
import matplotlib.pyplot as plt #导入Matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签

datafile = '../data/model.xls'
data = pd.read_excel(datafile)
data = data.values
shuffle(data)

p = 0.8 #设置训练数据比例
train = data[:int(len(data)*p),:]
test = data[int(len(data)*p):,:]
#构建LM神经网络模型
from keras.models import Sequential #导入神经网络初始化函数
from keras.layers.core import Dense, Activation #导入神经网络层函数、激活函数

netfile = '../tmp/net.model' #构建的神经网络模型存储路径

net = Sequential() #建立神经网络
#net.add(Dense(input_dim = 3, output_dim = 10)) #添加输入层（3节点）到隐藏层（10节点）的连接( Update your `Dense` call to the Keras 2 API: `Dense(input_dim=3, units=10)`)
net.add(Dense(input_dim = 3, units = 10)) #添加输入层（3节点）到隐藏层（10节点）的连接(
net.add(Activation('relu')) #隐藏层使用relu激活函数
#net.add(Dense(input_dim = 10, output_dim = 1)) #添加隐藏层（10节点）到输出层（1节点）的连接
net.add(Dense(input_dim = 10, units = 1)) #添加隐藏层（10节点）到输出层（1节点）的连接
net.add(Activation('sigmoid')) #输出层使用sigmoid激活函数
#net.compile(loss = 'binary_crossentropy', optimizer = 'adam', class_mode = "binary") #编译模型，使用adam方法求解
net.compile(loss = 'binary_crossentropy', optimizer = 'adam') #编译模型，使用adam方法求解
#net.fit(train[:,:3], train[:,3], nb_epoch=1000, batch_size=1) #训练模型，循环1000次(UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.)
net.fit(train[:,:3], train[:,3], epochs=1000, batch_size=1)
net.save_weights(netfile) #保存模型

#构建CART决策树模型
from sklearn.tree import DecisionTreeClassifier #导入决策树模型

treefile = '../tmp/tree.pkl' #模型输出名字
tree = DecisionTreeClassifier() #建立决策树模型
tree.fit(train[:,:3], train[:,3]) #训练

#保存模型
from sklearn.externals import joblib
joblib.dump(tree, treefile)

from sklearn.metrics import roc_curve  # 导入ROC曲线

predict_result = net.predict(test[:, :3]).reshape(len(test))  # 预测结果变形
fpr, tpr, thresholds = roc_curve(test[:, 3], predict_result, pos_label=1)
plt.plot(fpr, tpr, linewidth=2, label='LM神经网络ROC曲线')  # 作出LM的ROC曲线

fpr1, tpr1, thresholds1 = roc_curve(test[:, 3], tree.predict_proba(test[:, :3])[:, 1], pos_label=1)
plt.plot(fpr1, tpr1, linewidth=2, label='决策树ROC曲线')  # 做出ROC曲线
plt.xlabel('假正例率') #坐标轴标签
plt.ylabel('真正例率') #坐标轴标签
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)
plt.legend(loc=4)
plt.title('LM神经网络和决策树模型比较')
plt.savefig('dt_lm_roc.jpg')#pip install pillow
plt.show()