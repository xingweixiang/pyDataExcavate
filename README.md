# 说明
    本项目所有代码在本地调试通过
## 运行环境
     Flask python 3.7.*
## requirements
    Flask
    pip install numpy
    pip install pandas
    pip install matplotlib
    pip install mkl
    pip install scipy
    pip install statsmodels
    pip install xlrd
    pip install xlwt
    pip install scikit-learn --default-timeout=500
    pip install pywt
    pip install PyWavelets
    pip install keras --default-timeout=500
    pip install tensorflow --default-timeout=500
 ## 目录
* 数据分析与挖掘
	* [一、数据挖掘基础](#一数据挖掘基础)
		* [1、数据挖掘的基本任务](#1数据挖掘的基本任务)
		* [2、数据挖掘建模过程](#2数据挖掘建模过程)
	* [二、python数据分析工具](#二python数据分析工具)
		* [1、Numpy](#1Numpy)
		* [2、Scipy](#2Scipy)
		* [3、matplotlib](#3matplotlib)
		* [4、Pandas](#4Pandas)
		* [5、StatsModels](#5StatsModels)
		* [6、scikit-Learn](#6scikit-Learn)
		* [7、Keras](#7Keras)
		* [8、Gensim](#8Gensim)
		
## 一、数据挖掘基础

### 1、数据挖掘的基本任务
- 基本任务包含：分类和预测、聚类分析、关联规则、时序模式、偏差检验、智能推荐等。从数据中提取商业价值。
### 2、数据挖掘建模过程
- 定义挖掘的目标   
明确挖掘目标，明确完成效果。需要分析应用目标了解用户需求。
- 数据取样   
取样标准：相关性、可靠性、有效性。<br>不可忽视数据的质量问题，衡量数据质量的方式：<br>
资料的完整无缺，指标项齐全。<br>
数据的准确无误，为正常数据，不在异常指标状态的水平。<br>
数据抽样的方式：<br>
随机抽样<br>
等距抽样<br>
分层抽样<br>
从起始顺序抽样<br>
分类抽样<br>
- 数据探索<br>
对样本数据的提前探索和预处理是保证模型质量的重要条件。<br>
数据探索主要包括：异常值分析、缺失值分析、相关分析、周期性分析等。
- 数据预处理<br>
采集数据过大的时候，怎么降维处理，怎么做缺失值处理都是使用数据预处理来完成的。采样的数据中同样会有噪声，不完整，不一致的数据也需要做预处理准备。<br>
数据预处理主要包括：数据筛选、数据变量转换、缺失值处理、坏数据处理、数据标准化、主成分分析、属性选择、数据约规等。
- 挖掘建模<br>
完成与处理后，思考问题：本次建模是属于数据挖掘中的哪一类(分类、聚类、关联规则、时序模式或者智能推荐)，是要使用什么样的算法来构建模型？这一步是核心关键。
- 模型评价<br>
从建模之后得出来一些分析结果，从而找到一个最好的模型。做出根据业务对模型的解释和应用。
## 二、[python数据分析工具](/example/chapter2/demo/code)
numpy、scipy、matplotlib、pandas、statsModels、scikit-Learn、Kears、Gensim、Pillow(原PIL)
### 1、numpy
- 提供多维数组支持，以及相应的高效的处理函数<br>
基本操作：<br>
创建数组  a = np.array([2, 0, 1, 5])<br>
引用前三个数字（切片）a[:3]<br>
最大最小值  a.max() a.min()<br>
从小到大排序，此操作直接修改原数组   a.sort()<br>
Numpy是Python中相当成熟和常用的库，教程多，最值得看的是它官网帮助文档。<br>
参考链接：<br>
[http://www.numpy.org/](http://www.numpy.org/)<br>
[http://reverland.org/python/2012/08/22/numpy](http://reverland.org/python/2012/08/22/numpy/)
### 2、Scipy
- 提供矩阵支持，以及矩阵相关的数值计算模块。功能包含有最优化、线性代数、积分、插值、拟合、特殊函数、快速傅里叶变换、信号处理、图像处理、常微分方程求解<br>
Scipy依赖Numpy<br>
参考链接：<br>
[https://www.scipy.org/](https://www.scipy.org/)<br>
[http://reverland.org/python/2012/08/24/scipy](http://reverland.org/python/2012/08/24/scipy/)
### 3、matplotlib
- 提供强大的数据可视化工具、作图库，主要用于二维绘图。<br>
```
    #中文不显示的问题
    plt.reParams['font.sans-serif'] = ['SimHei']    # 如果中文字体是SimHei
    #负号显示为方块的问题
    plt.rcParams['axes.unicode_minus']= False
```
参考链接：<br>
[http://matplotlib.org/](http://matplotlib.org/)<br>
[http://reverland.org/python/2012/09/07/matplotlib-tutorial](http://reverland.org/python/2012/09/07/matplotlib-tutorial)<br>
[http://matplotlib.org/gallery.html](http://matplotlib.org/gallery.html)（画廊）
### 4、Pandas
- 张大灵活的数据分析和探索工具，支持类似SQL的数据增、删、查、改，支持时间序列分析、缺失数据处理等。基本的数据结构是 Series和DataFrame<br>
着眼于数据的读取、处理和探索
```
1. Series：
    序列  类似一维数组  
    有index用于定位元素，index可以为数字，也可以为其他，类似主键
    创建序列  s = pd.Series([1,2,3],index = ['a','b','c'])

2. DataFrame：
    表格  类似二维数组
    每一列都是一个series   本质是series的容器
    每个series有唯一的表头，用于区分  多个series的index相同
    创建表  d = pd.DataFrame([1,2,3],[4,5,6],colume = ['a','b','c'])
    也可以直接  d = pd.DataFrame(s)
    d.head()
    d.describe()
    pd.read_excel('filename')
    pd.read_csv('filename',encoding = 'utf-8)
    补充操作
    pd.notnull(x)  得到x的不为空的true和false
    x[pd.notnull(x)]  可得到x中不为空的项  list的话只能根据int来进行索引 series可以通过true和false
    map接收一个序列 list或者numpy的array   
    dataFrame的排序 dataFrame.sort_values(['confidence','support'], ascending = False)  
    可用dataFrame[[index1,index2]] 访问  
```
参考链接：<br>
[http://pandas.pydata.org/pandas-docs/stable](http://pandas.pydata.org/pandas-docs/stable/)<br>
[http://jingyan.baidu.com/season/43456](http://jingyan.baidu.com/season/43456/)
### 5、StatsModels
- 统计建模和计量经济学，包括描述统计、统计模型估计和推断。着眼于数据的统计建模分析，支持与Pandas进行数据交互<br>
依赖于Pandas和pasty(描述统计的库)<br>
可进行ADF平稳性检验、白噪声检验等<br>
参考链接：<br>
[http://www.statsmodels.org/stable/index.html](http://www.statsmodels.org/stable/index.html/)
### 6、scikit-Learn
- 强大的机器学习相关库，提供完整的机器学习工具箱，包括数据预处理、分类、回归、聚类、预测和模型分析等<br>
依赖于Numpy、SciPy、Matplotlib
```
1. 所有模型的接口：
    model.fit()：训练数据  fit（X,y）——监督学习    fit（X）——非监督学习
2. 监督学习
    model.predict(x_new) 预测新样本
    model.predict_proba(x_new) 预测概率
    model.score() 得分越高越好
3. 非监督学习
    model.transform()  从数据中学习到新的“基空间”
    model.fit_transform()  从数据中学到新的基并将这个数据按照这组基进行转换
```
参考链接：<br>
[http://scikit-learn.org](http://scikit-learn.org/)
### 7、Keras
- 深度学习库，用于建立神经网络以及深度学习模型。人工神经网络  基于Theano<br>
不仅可以搭建普通的神经网络，还可以搭建各种深度学习模型，如自编码器、循环神经网络、递归神经网络、卷积神经网络等<br>
依赖于Numpy、Scipy和Theano
```
from keras.models import Sequential
from keras.layers.core import Dense, Activation
model = Sequential() #建立模型
model.add(Dense(input_dim = 3, output_dim = 10))
model.add(Activation('relu')) #用relu函数作为激活函数，能够大幅提供准确度
model.add(Dense(input_dim = 10, output_dim = 1))
model.add(Activation('sigmoid')) #由于是0-1输出，用sigmoid函数作为激活函数
model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
#编译模型。由于我们做的是二元分类，所以我们指定损失函数为binary_crossentropy，以及模式为binary
#另外常见的损失函数还有mean_squared_error、categorical_crossentropy等，请阅读帮助文件。
#求解方法我们指定用adam，还有sgd、rmsprop等可选
model.fit(x, y, nb_epoch = 1000, batch_size = 10) #训练模型，学习一千次
yp = model.predict_classes(x).reshape(len(y)) #分类预测
#model.predict()给出概率
#model.predict_classes()给出分类结果
```
参考链接：<br>
[http://deeplearning.net/software/theano/install.html#install](http://deeplearning.net/software/theano/install.html#install/)<br>
[https://github.com/fchollet/keras](https://github.com/fchollet/keras/)
### 8、Gensim
- 用来做文本主题模型的库，文本挖掘，处理语言方面的任务，如文本相似度计算，LDA，Word2Vec
参考链接：<br>
[http://radimrehurek.com/gensim](http://radimrehurek.com/gensim/)<br>