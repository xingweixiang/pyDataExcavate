# 说明
    本项目所有代码在本地调试通过
## 运行环境
     python 3.7.*
## requirements
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
    pip install pillow 
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
	* [三、数据探索](#三数据探索)
		* [1、数据质量分析](#1数据质量分析)
		* [2、数据特征分析](#2数据特征分析)
		* [3、数据探索主要函数](#3数据探索主要函数)
    * [四、数据预处理](#四数据预处理)
        * [1、数据清洗](#1数据清洗)
        * [2、数据集成](#2数据集成)
        * [3、数据变换](#3数据变换)
        * [4、数据规约](#4数据规约)
        * [5、预处理主要函数](#5预处理主要函数)
    * [五、挖掘建模](#五挖掘建模)
        * [1、分类与预测](#1分类与预测)
        * [2、聚类分析](#2聚类分析)
        * [3、关联规则](#3关联规则)
        * [4、时序模式](#4时序模式)
        * [5、离群点检测](#5离群点检测)
    * [六、电力窃漏电用户自动识别](#六电力窃漏电用户自动识别)
        * [1、背景与挖掘目标](#1背景与挖掘目标)
        * [2、数据探索分析及数据预处理](#2数据探索分析及数据预处理)
        * [3、模型构建](#3模型构建)
    * [七、航空公司客户价值分析](#七航空公司客户价值分析)
        * [7-1、背景与挖掘目标](#7-1背景与挖掘目标)
        * [7-2、数据探索分析及数据预处理](#7-2数据探索分析及数据预处理)
        * [7-3、模型构建](#7-3模型构建)
    * [八、中医证型关联规则挖掘](#八中医证型关联规则挖掘)
        * [8-1、背景与挖掘目标](#8-1背景与挖掘目标)
        * [8-2、数据探索分析及数据预处理](#8-2数据探索分析及数据预处理)
        * [8-3、模型构建](#8-3模型构建)
    * [九、基于水色图像的水质评价](#九基于水色图像的水质评价)
        * [9-1、背景与挖掘目标](#9-1背景与挖掘目标)
        * [9-2、数据探索分析及数据预处理](#9-2数据探索分析及数据预处理)
        * [9-3、模型构建](#9-3模型构建)
    * [十、家用电器用户行为分析与事件识别](#十家用电器用户行为分析与事件识别)
        * [10-1、背景与挖掘目标](#10-1背景与挖掘目标)
        * [10-2、数据探索分析及数据预处理](#10-2数据探索分析及数据预处理)
        * [10-3、模型构建](#10-3模型构建)
        * [10-4、模型检验](#10-4模型检验)
    * [十一、应用系统负载分析与磁盘容量预测](#十一应用系统负载分析与磁盘容量预测)
        * [11-1、背景与挖掘目标](#11-1背景与挖掘目标)
        * [11-2、数据探索分析及数据预处理](#11-2数据探索分析及数据预处理)
        * [11-3、模型构建](#11-3模型构建)
        * [11-4、模型评价](#11-4模型评价)
    * [十二、电子商务网站用户行为分析及服务推荐](#十二电子商务网站用户行为分析及服务推荐)
        * [12-1、背景与挖掘目标](#12-1背景与挖掘目标)
        * [12-2、数据探索分析及数据预处理](#12-2数据探索分析及数据预处理)
        * [12-3、模型构建](#12-3模型构建)
    * [十三、财政收入影响因素分析及预测模型](#十三财政收入影响因素分析及预测模型)
        * [13-1、背景与挖掘目标](#13-1背景与挖掘目标)
        * [13-2、数据探索分析及数据预处理](#13-2数据探索分析及数据预处理)
        * [13-3、模型构建](#13-3模型构建)
    * [十四、基于基站定位数据的商圈分析](#十四基于基站定位数据的商圈分析)
        * [14-1、背景与挖掘目标](#14-1背景与挖掘目标)
        * [14-2、数据探索分析及数据预处理](#14-2数据探索分析及数据预处理)
        * [14-3、模型构建](#14-3模型构建)
    * [十五、电商产品评论数据情感分析](#十五电商产品评论数据情感分析)
        * [15-1、背景与挖掘目标](#15-1背景与挖掘目标)
        * [15-2、数据探索分析及数据预处理](#15-2数据探索分析及数据预处理)
        * [15-3、模型构建](#15-3模型构建)
    * [十六、企业偷漏税识别模型](#十六企业偷漏税识别模型)
        * [16-1、背景与挖掘目标](#16-1背景与挖掘目标)
        * [16-2、数据探索分析及数据预处理](#16-2数据探索分析及数据预处理)
        * [16-3、模型构建](#16-3模型构建)
        * [16-4、模型评价](#16-4模型评价)
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
[http://radimrehurek.com/gensim](http://radimrehurek.com/gensim/)
## 三、[数据探索](/example/chapter3/demo)
通过检验数据集的数据质量、绘制图表、计算特征量等手段，对样本数据集的结构和规律进行分析的过程就是数据探索。数据探索有助于选择合适的数据预处理和建模方法，甚至可以完成一些通常由数据挖掘解决的问题。
### 1、数据质量分析
主要任务是检查原始数据中是否存在脏数据，包括缺失值，异常值，不一致值，重复数据及特殊符号数据
- 缺失值分析：包括记录缺失和记录的某字段缺失等<br>
产生原因：无法获取、遗漏、属性值不存在；<br>
影响：有用信息缺乏、不确定性加重、不可靠<br>
处理：删除、补全、不处理
- 异常值，不合常理的数据，剔除可消除不良影响，分析可进行改进。异常值分析也称离群点分析。
常用的分析方法：简单统计量分析(如max、min)；3σ原则(99.7%)；箱型图<br>
- 一致性分析：直属局矛盾性、不相容性
产生原因：数据集成过程中，数据来自不同数据源，存放等未能进行一致性更新
### 2、数据特征分析
- 分布分析：数据分布特征与分布类型<br>
定量数据分布分析：一般按以下步骤，求极差—>决定组距和组数—>决定分点—>列频率分布表—>绘频率分布直方图<br>
定性数据分布分析：采用分类类型来分组，用饼图或条形图来描述分布
- 对比分析：两个指标进行比较，展示说明大小水平高低，速度快慢，是否协调等。
绝对数比较：是利用绝对数进行对比，从而寻找差异的一种方法<br>
相对数比较：结构相对数(比重)，比例相对数(比值)，比较相对数(同类不同背景)，强度相对数(密度)，计划完成程度相对数，动态相对数
- 统计量分析：用统计指标对定量数量进行统计描述
集中趋势：均值、中位数、众数<br>
离中趋势：极差、标准差、变异系数(CV=标准差/平均值*100%)、四分位数间距(上下四分位数之差)
周期性分析：是探索变量是否随时间呈周期变化趋势
贡献度分析：又称帕累托分析，原理是帕累托法则，又称20/80定律。同样的投入在不同的地方产生不同的收益
```
#-*- coding: utf-8 -*-
#菜品盈利数据 帕累托图
from __future__ import print_function
import pandas as pd

#初始化参数
dish_profit = '../data/catering_dish_profit.xls' #餐饮菜品盈利数据
data = pd.read_excel(dish_profit, index_col = u'菜品名')
print(data)
data = data[u'盈利'].copy()
print('------------------')
print(data)
#data.sort(ascending = False) 会提示AttributeError: 'Series' object has no attribute 'sort'也就是说Series没有sorted这个方法 应该这样：sorted(.....) ...是你要排序的Seri
data.sort_values(ascending = False) #可以用Series.sort_values()方法,对Series值进行排序。
import matplotlib.pyplot as plt #导入图像库
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

plt.figure()
plt.title('帕累托图');
data.plot(kind='bar')
plt.ylabel(u'盈利（元）')
p = 1.0*data.cumsum()/data.sum()
p.plot(color = 'r', secondary_y = True, style = '-o',linewidth = 2)
plt.annotate(format(p[6], '.4%'), xy = (6, p[6]), xytext=(6*0.9, p[6]*0.9), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")) #添加注释，即85%处的标记。这里包括了指定箭头样式。
plt.ylabel(u'盈利（比例）')
plt.show()
```
![帕累托图](/example/chapter3/demo/data/plt.jpg)
- 相关性分析：分析连续变量之间线性相关程度的强弱，并用适当的统计指标表示出来 
直接绘制散点图<br>
绘制散点图矩阵，对多个变量两两关系的散点图<br>
计算相关系数
### 3、数据探索主要函数
- 主要是Pandas用于数据分析和Matplotlib用于数据可视化<br>
Pandas主要统计特征函数<br>
sum 计算数据样本的总和(按列)<br>
mean 计算数据样本的算数平均值<br>
var 计算数据样本的方差<br>
std 标准差<br>
corr 计算数据样本的Spearman/Pearson相关系数矩阵<br>
cov 计算数据样本的协方差矩阵<br>
skew 样本值的偏度(三阶矩)<br>
kurt 样本值的峰度(四阶矩)<br>
describe() 给出样本的基本描述(如均值、标准差等)<br>
- Pandas累积统计特征函数<br>
cumsum 依次给出前1-n个数的和<br>
cumprod 依次给出前1-n个数的积<br>
cummax 依次给出前1-n个数的最大值<br>
cummin 依次给出前1-n个数的最小值
- 统计作图函数，基于Matplotlib<br>
plot 绘制线性二维图，折线图<br>
pie 绘制饼图<br>
hist 绘制二维条形直方图<br>
boxplot 绘制箱型图 Pandas<br>
plot(logy=True) 绘制y轴的对数图形 Pandas<br>
plot(yerr=error) 绘制误差条形图 Pandas
## 四、[数据预处理](/example/chapter4/demo)
数据预处理：主要包括数据清洗，数据集成，数据变换和数据规约
### 1、数据清洗
- 主要是删除原始数据集中的无关数，重复数据，平滑噪声数据，筛选掉与挖掘主题无关的数据，处理缺失值，异常值等<br>
缺失值的处理：删除记录，数据插补和不处理三种方法。<br>
数据插补方法：<br>
1、均值、中位数、众数插补；<br>
2、使用固定值插补：将缺失的属性值用一个常量进行替换；<br>
3、最近临插补：在记录中找到与缺失样本最接近的样本的该属性值进行插补；<br>
4、回归方法：对带有缺失值的变量，根据已有数据和与其有关的其他变量的数据建立拟合模型老预测确实的属性值；<br>
5、插值法：利用已知点建立合适的插值函数，未知值由丢in给点求出的函数值近似代替。
插值方法：<br>
1、拉格朗日插值法：但是在实际计算中很不方便。<br>
2、牛顿插值法：具有承袭性和易于变动节点的特点。<br>
在python的scipy库中只提供了拉格朗日插值法的函数，牛顿法需要自己进行编写
- 异常值处理：在数据预处理时，异常值是否剔除，要视情况而定。<br>
 1、删除含有异常值的记录；<br>
 2、视为缺失值：利用缺失值的方法进行处理；<br>
 3、平均值修正：可用前后两个观测值的平均值修正改异常值；<br>
 4、不处理：直接在具有异常值的数据上进行挖掘建模
### 2、数据集成
- 数据往往分布在不同的数据源中，数据集成就是将多个数据源合并存放在一个一致的数据存储中的过程。<br>
1、实体识别：从不同数据源识别出现实世界的实体，统一不同源数据的矛盾之处；<br>
2、冗余属性识别：数据集成往往导致数据冗余：统一属性多次出现；同一属性命名不一致导致重复。对于冗余数据先分析，检测后进行删除。
### 3、数据变换
- 主要是对数据进行规范化处理，将数据转换成适当的形式，以适用于挖掘任务及算法的需要。<br>
1、简单函数变换：是对原始数据进行某些数学函数变换，常用的包括平方、开方、取对数、差分运算等。简单的函数变换常用来将不具有正态分布的数据变换成具有正态分布的数据。
在时间序列分析中，简单的对数变换或者差分运算可以将非平稳序列转换成平稳序列。<br>
2、规范化：为了消除指标间的量纲和取值范围差异的影响，需要进行标准化处理，将数据按照比例进行缩放。<br>
最小-最大规范化：也称为离差标准化，是对原始数据进行线性变换，将数值映射到[0-1]之间，x*=(x-min)/(max-min)<br>
零-均值规范化：也称标准差标准化，经过处理的数据均值为0，标准差为1，x*=(x-平均值)/标准差<br>
小数定标规范化：通过移动属性值的小数位数，将属性值映射到[-1,1]之间，移动的小数位数取决于属性值绝对值的最大值<br>
### 4、数据规约
- 在大数据上进行复杂的数据分析和挖掘需要很长时间，数据规约产生更小但保持原数据完整性的新数据集。在规约后的数据集上分析和挖掘更有效率。
### 5、预处理主要函数
- 主要是插值、数据归一化、主要分分析等与数据预处理相关的函数<br>
interpolate 一维、高维数据插值 Scipy<br>
unique 去除重复元素、得到单值列表 Pandas/Numpy<br>
isnull 判断是否空值 Pandas<br>
notnull 判断是否非空值 Pandas<br>
PCA 对指标变量矩阵进行主成分分析 Scikit-Learn<br>
random 生成随机矩阵 Numpy<br>
## 五、[挖掘建模](/example/chapter5/demo)
经过数据探索和数据预处理，得到了可以直接建模的数据。根据挖掘目标和数据形式可以建立分类与预测、聚类分析、关联规则、时序模式和偏差检测等模型。
### 1、分类与预测
- 分类：是一个构造分类模型，输入样本的属性值，输出对应的类别，将每个样本映射到先定义好的类别；分类模型建立在已有类标记的数据集上，模型在已有样本上的准确率可以方便计算。<br>
分类两步：<br>第一步是学习步，通过归纳分析训练样本集哎加你分诶模型得到分类规则；<br>第二步是分类步，先用一直的测试样本集评估分类规则的准确率，如果准确率可以接受，则使用该模型对未知类标号的待测样本集进行预测。<br>
预测：是指建立两种或两种以上变量间相互依赖的函数模型，然后进行预测或控制。
预测两步：<br>第一步是通过训练集建立预测属性的函数模型，<br>第二步在模型通过检验后进行预测或控制。<br>
常用的分类与回归算法：<br>
1、回归分析：通过建立模型来研究变量之间相互关系的密切程度，结构状态及进行模型预测的一种有效工具。<br>回归分析研究内容包括：<br>
线性回归（一元线性回归，多元线性回归，多个自变量与多个因变量的回归）；<br>
回归诊断（如何从数据推断回归模型基本假设的合理性，基本假设不成立时如何对数据进行修正，判断回归方程拟合的效果，选择回归函数的形式）<br>
回归变量选择（自变量选择的标准，逐步回归分析法）<br>
参数估计方法改进（偏最小二乘回归，岭回归，主成分回归）<br>
非线性回归（一元非线性回归，分段回归，多元非线性回归）<br>
含有定性变量的回归（自变量含有定性变量的情况，因变量含有定性变量的情况）。<br>
（1）线性回归：自变量与因变量之间是线性关系，可以用最小二乘法求解模型系数<br>
（2）非线性回归：因变量与自变量之间不都是线性关系，如果非线性关系可以通过简单的函数变换化成线性关系，用线性回归的思想求解，如果不能转化，用非线性最小二乘法求解<br>
（3）logistic回归：因变量一般有0和1两种取值。是广义线性回归模型的特例，利用logistic函数将因变量的取值范围控制在0和1之间，表示取值为 的概率。<br>
logistic回归模型是建立ln(p/1-p)与自变量的线性回归模型，即ln(p/1-p)=b0+b1x1+b2x2+...+bnxn。<br>
logistic回归建模步骤:<br>
1）、根据分析目的设置指标变量，收集数据，根据收集到的数据对特征进行筛选，特征筛选的方法有很多，主要包含在scikit-learn的feature_selection库中，选择F值较大或者P值较小的特征。其次还有递归特征消除和稳定性选择等方法；<br>
2）、列出线性回归方程，估计出模型中的系数；<br>
3、进行模型检验，有正确率，混淆矩阵，ROC曲线，KS值等；<br>
4、模型应用，输入自变量的值可以得到因变量的值<br>
（4）岭回归：参与建模的自变量之间具有多重共线性，是一种改进的最小二乘估计<br>
（5）主成分分析：参与建模的自变量之间具有多重共线性，是参数估计的一种有偏估计，可以消除多重共线性<br>
2、决策树算法分类：<br>
1、ID3算法：其核心是在决策树的各级节点上，使用信息增益方法作为属性的选择标准，来帮助确定生成每个节点时所应采用的合适属性<br>
2、C4.5算法：是使用信息增益率来选择节点属性，ID3只适用于离散的属性描述，而C4.5既能够处理离散的描述属性，也可以处理连续的描述属性<br>
3、CART算法：是一种十分有效地非参数分类和和i回归方法，通过构建树、修剪树、评估树来构建一个二叉树，当终节点是连续变量时，该树为回归树，当终节点是分类变量时，该树为分类树
ID3计算步骤：<br>
（1）对当前样本集合，计算所有属性的信息增益；<br>
（2）训责信息增益最大的属性作为测试属性，把测试属性取值相同的样本划为同一子样本集；<br>
（3）若子样本集的类别属性只含有单个属性，则分支为叶子节点，判断其属性值并标上相应的符号，然后返回调用处；否则对子样本集递归调用本算法。<br>
3、人工神经网络：是模拟生物神经网络进行信息处理的一种数学模型。人工神经元是人工神经网络操作的基本信息处理单位。
人工神经网络的学习也称为训练，指的是神经网络在收到外部环境的刺激下调整神经网络的参数，使神经网络以一种新的方式对外部环境做出反应的一个过程。
在分类与预测中，人工神经网络主要使用指导的学习方式，即根据给定的训练样本，调整人工神经网络的参数以使网络输出接近于已知的样本类标记或其他形式的因变量。
在人工神经网络的发展过程中，提出了多种不同的学习规则，没有一种特定的学习算法适用于所有的网络结构和具体问题。在分类与预测中，西格玛学习规则（误差矫正学习算法）是使用最广泛的一种。
误差校正想学习算法根据神经网络的输出误差对神经元的连接强度进行修正，属于指导学习。
神经网络训练是否完成常用误差函数E来衡量，当误差函数小于某一个设定的值时即停止神经网络的训练。
使用人工神经网络模型需要确定网络连接的拓扑结构、神经元的特征和学习规则等。常用来实现分类和预测的人工神经网络算法如下：<br>
1、BP神经网络：是一种按误差逆传播算法训练的多层前馈网络，学习算法是西格玛学习规则，是目前 应用最广泛的神经网络模型之一。<br>
2、LM神经网络：是基于梯度下降法和牛顿法结合的多层前馈网络，迭代次数少，收敛速度快，精确度高<br>
3、RBF径向基神经网络：RBF网络能够以任意精度逼近任意连续函数，从输入层到隐含层的变换是非线性的，而从隐含层到输出层的变换是线性的，特别适合解决分类问题。<br>
4、FNN模糊神经网络：FNN模糊神经网络是具有模糊权系数或者输入信号是模糊量的神经网络，是模糊系统与神经网络相结合的产物，它汇聚了神经网络与模糊系统的优点，集联想、识别、自适应及模糊信息处理于一体。<br>
5、GMDH神经网络：也称为多项式网络，是前馈神经网络中常用的一种用于预测的神经网络，特点是网络结构不固定，而且在训练过程中不断改变。<br>
6、ANFIS自适应神经网络：神经网络镶嵌在一个全部模糊的结构之中，在不知不觉中向训练数据学习，自动产生、修正并高度概括出最佳的输入与输出变量的隶属函数以及模糊规则；另外，神经网络的各层结构与参数也都具有了明确的、易于理解的物理意义。
BP算法的学学习过程由信号的正向传播与误差的逆向传播两个过程组成。正向传播时，输入信号经过隐层的处理后，传向输入层。若输出层节点未能得到期望的输出，则转入误差的逆向传播阶段，将输出误差按某种子形式，通过隐层向输入层返回，
从而获得各层单元的参考误差或称误差信号，作为修改各单元权值的依据。此过程一直进行到网络输出的误差逐渐减少到可接受的程度或达到设定的学习次数为止。<br>
4、贝叶斯网络：贝叶斯网络又称信度网络，是Bayes方法的扩展，是目前不确定知识表达和推理领域最有效的理论模型之一。<br>
5、支持向量机：支持向量机是一种通过某种非线性映射，把低维的非线性可分转化为高维的线性可分，在高维空间进行线性分析的算法。
 ### 2、聚类分析
- 聚类分析是在没有给定划分类别的情况下，根据数据相似度进行样本分组的一种方法。可以建立在无类标记的数据上，是一种非监督的学习算法。划分原则是组内距离最小化，组间距离最大化。<br>
常用的聚类方法：<br>
1、划分方法：K-Means（K均值），K-Medoids（K-中心点），Clarans算法<br>
2、层次分析方法：BIRCH算法（平衡迭代规约和聚类），CURE算法（代表点聚类），CHAMELEON算法（动态模型）。<br>
3、基于密度的方法：DBSCAN算法，DENCLUE算法，OPTICS算法<br>
4、基于网格的方法：STING算法，CLIOUE算法，WACE-CLISTER算法<br>
5、基于模型的方法：统计学方法，神经网络方法。<br>
常用聚类分析算法：<br>
1、K-MEANS：K-均值聚类也称快速聚类法，在最小化误差函数的基础上将数据划分为预定的类数K。原理简单，便于处理大量数据<br>
2、K-中心点：K-均值对于孤立点敏感。K-中心点算法不采用簇中对象的平均值作为簇中心，而选用簇中离平均值最近的对象作为簇中心。<br>
3、系统聚类：也称多层次聚类，分类的单位由高到低呈树形结构，且所处的位置越低，其所包含的对象就越少，但这些对象间的共同特征越多。只适合在小数据量的时候使用，数据量大的时候速度会非常慢。
K-均值聚类算法：是基于距离的非层次聚类算法，在最小化误差函数的基础上将数据划分为预定的类数K，采用距离作为相似性的评价指标。<br>
算法过程：<br>
1)、从N个样本数据中随机选取K个对象作为初始的聚类中心<br>
2)、分别计算每个样本到各个聚类中心的距离，将对象分配到距离最近的聚类中<br>
3)、所有对象分配完成后，重新计算K个聚类的中心<br>
4)、与前一次计算得到的K个聚类中心比较，如果聚类中心发生变化，转到过程2，否则转到过程5，<br>
5)、当质心不发生变化时停止并输出聚类结果。<br>
数据类型与相似性的度量：<br>
1)、连续属性：先对各属性值进行零-均值规范，在进行距离计算。一般需要度量样本之间的距离，样本与簇之间的距离，簇与簇之间的距离。<br>
样本之间距离常用欧几里得距离、曼哈顿距离和闵科夫斯基距离；样本与簇之间距离用簇中心到样本的距离；簇与簇之间距离用簇中心的距离。<br>
2)、文档数据：对于文档数据使用余弦相似性度量，先将文档数据整理成文档-词矩阵格式，两个文档之间相似度用d(i,j)=cos(i,j)表示<br>
目标函数：使用误差平方和SSE作为度量聚类质量的目标函数，对于两种不同的聚类结果，选择误差平方和较小的分类结果。
### 3、关联规则
- 关联规则也称购物篮分析。<br>
常用关联规则算法：<br>
1、Apriori：关联规则最常用的挖掘频繁项集的算法，核心思想是通过连接产生选项及其支持度然后通过剪枝生成频繁项集。<br>
2、FP-Tree：针对Apriori固有的多次扫描事务数据集的缺项，提出不产生候选频繁项集的方法<br>
3、Eclat：是一种深度优先算法，采用垂直数据表示形式，在概念格理论的基础上利用基于前缀的等价关系将搜索空间划分为较小的子空间<br>
4、灰色关联法：分析和确定各因素之间的影响程度或是若干子因素对主因素的贡献度进行分析的方法
### 4、时序模式
- Python实现时序模式的主要库是StatsModels（能用Pandas，就用Pandas先做），算法主要是ARIMA模型。
- 常用模型：平滑法、趋势你合法、组合模型、AR模型、MA模型、ARMA模型、ARIMA、ARCH、GARCH模型及衍生。
- python主要时序算法函数：acf自相关，plot_acf画自相关系数图、pacf计算偏相关系数、plot_pacf画偏相关系数图、adfuller对观测值序列进行单位根检验、diff差分计算、ARIMA创建ARIMA时序模型、summary或summaty2给出ARIMA模型报告、aic/bic/hqic计算ARIMA模型的指标值、forecast预测、acorr_ljungbox检验白噪声
### 5、离群点检测
- 离群点检测：发现与大部分其他对象显著不同的对象<br>
离群点成因：数据来源于不同的类，自然变异，数据测量和收集误差<br>
离群点类型：<br>
1、全局离群点和局部离群点：从整体来看某些对象没有离群特征，但是从局部来看，却显示了一定的离群性。<br>
2、数值型离群点和分类型离群点<br>
3、一维离群点和多维离群点<br>
离群点检测方法：<br>
1、基于统计：构建一个分布模型，并计算对象符合该模型的概率，把具有低概率的对象视为离群点，前提是必须知道数据集服从什么分布<br>
2、基于近邻度：在数据对象之间定义邻近性度量，把远离大部分点的对象视为离群点<br>
3、基于密度：数据集可能存在不同密度区域，离群点是在低密度区域中的对象，一个对象的离群点得分是该对象周围密度的逆<br>
4、基于聚类：丢弃远离其他簇的小簇；或者先聚类所有对象，然后评估对象属于簇的程度。<br>
基于模型的离群点检测方法：通过估计概率分布的参数来建立一个数据模型。如果一个数据对象不能很好地同该模型拟合，即如果它很可能不服从该分布，则是一个离群点<br>
1、一元正态分布中的离群点检测：N（0，1）的数据对象出现在该分布的两边尾部的机会很小，因此可以用它作为检测数据对象是否是离群点的基础，数据对象落在3倍标准差中心区域之外的概率仅有0.0027<br>
2、混合模型的离群点检测：混合模型是一种特殊的统计模型，它使用若干统计分布对数据建模，每个分布对应一个簇，而每个分布的参数提供对应簇的描述，通常用中心和发散描述<br>
3、基于聚类的离群点检测方法：<br>
（1）丢弃远离其他簇的小簇：该过程可以简化为丢弃小于某个阈值的所有簇<br>
（2）基于原型的聚类：首先聚类所有对象，然后评估对象属于簇的程度。可以用对象到他的簇中心的距离来度量属于簇的程度。<br>
基于原型的聚类主要有两种方法评估对象属于簇的程度：<br>
意识度量对象到簇原型的距离，并用它作为该对象的离群点得分；<br>
考虑到簇具有不同的密度，可以度量簇到原型的相对距离，相对距离是点到质心的距离与簇中所有点到质心的距离的中位数之比
## 六、[电力窃漏电用户自动识别](/example/chapter6/demo)
### 1、背景与挖掘目标
- 通过电力系统采集到的数据，提取出窃漏电用户的关键特征，构建窃漏电用户的识别模型。利用实时监测数据，调用窃漏电用户识别模型实现实时诊断，以实现自动检查、判断用户是否是存在窃漏电行为。
### 2、数据探索分析及数据预处理
- 数据抽取：抽取与窃漏电相关的用电负荷数据、终端报警数据、违约窃漏电罚信息以及用户档案资料等数据
- 数据探索分析：本案例采用分布分析和周期性分析等方法对电量数据进行数据分析<br>
（1）对一段时间所有窃漏电用户进行分布分析，从用电类别窃漏电情况图，得到非居民类别不存在窃漏电情况，故分析中不考虑非居民类别的用电数据。<br>
（2）随机抽取一个正常用电用户和一个窃漏电用户，采用周期性分析对用电量进行探索。分析出正常用电到窃漏电过程是用电量持续下降的过程，用户用电量开始下降，并且持续下降，就是用户开始窃漏电时所表现出来的重要特征。
- 数据预处理：本案例主要从数据清洗、缺失值处理和数据变换等方面对数据进行预处理<br>
数据清洗：主要目的是筛选出需要的数据，由于原始数据并不是所有的都需要进行分析，故处理时，需过滤。<br>
缺失值处理：在用户电量抽取过程中，发现存在缺失的现象。若抛弃掉，会影响供出电量的计算结果，导致日线损率误差很大。为了达到较好的建模效果，需对缺失值处理。本案例采用拉格朗日插值法对缺失进行插补。<br>
数据变换：原始数据虽然在一定程度上能反映用户窃漏电行为，但要作为构建模型的专家样本，特征不明显，需要进行重新构造。基于数据变换，得到新的评价指标来表征窃漏电行为具有的规律。<br>
构建专家样本：通过以上的数据变换得到的特征选取样本数据，得到专家样本库。
### 3、模型构建
- 构建窃漏电用户识别模型：窃漏电用户识别可通过构建分类预测模型来实现，常用的分类预测模型有LM神经网络和CART决策树，各个模型有各优点。故采用两种方法构建，并从中选择最优的分类模型。
- 模型评价：对于训练样本，LM神经网络和CART决策树的分类准确率相关不大，为了进一步评模型分类的性能，故利用测试样本对两模型进行评价，采用ROC曲线评价方法进行评估，一个优秀分类器对应的ROC曲线应该是尽量靠近左上角的。经过对比发现LM神经网络的ROC曲线比CART决策树的ROC曲线更加靠近左上角，说明LM神经网络模型的分类性能较好，能应用于构建窃漏电用户识别。<br>
```
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
```
![模型对比ROC曲线](/example/chapter6/demo/code/dt_lm_roc.jpg)
- 进行窃漏电诊断：在线监测用户用电负荷及终端报警数据，并经过处理，得到模型输入数据，利用构建好的窃漏电用户识别模型计算用户的窃漏电诊断结果，实现窃漏电用户实时诊断，并与实际稽查结果作对比。
## 七、[航空公司客户价值分析](/example/chapter7/demo)
### 7-1、背景与挖掘目标
- 根据航空公司提供的数据，对其客户进行分类。对不同的客户类别进行特征分析，并且比较不同类别客户的价值。对不同价值的客户类别提供个性化服务，制定相应的营销策略。
### 7-2、数据探索分析及数据预处理
- 数据抽取：以2014-03-31为结束时间，选取宽度为两年的时间段作为分析观测窗口，抽取所有客户的乘机记录形成历史数据。后续新增的客户采用同样的方法进行抽取，形成增量数据。
- 数据探索分析：本案例对数据进行缺失值分析，分析出数据的规律以及异常值。<br>
- 数据预处理：本案例主要采用数据清洗、属性规约和数据变换的预处理方法<br>
数据清洗：通过数据探索分析，发现数据中存在缺失值，票价最小值为0、折扣率最小值为0、总飞行公里数大于0的记录。由于原始数据量大，这类数据所占比例小，对问题影响不大，因此对其进行丢弃处理。(用Pandas处理)<br>
属性规约：原始数据中属性太多，删除与其不相关、弱相关或冗余的属性。<br>
数据变换：将数据转换，以适应挖掘任务及算法。本例主要采用属性构造和数据标准化的数据变换。
### 7-3、模型构建
根据航空公司客户5个指标的数据，对客户进行聚类分群；结合业务对每个客户群进行特征分析，分析其客户价值，并对每个客户群进行排名
- 客户聚类：采用K-Means聚类算法对客户进行客户分群，聚成5类，进行客户价值分析。
```
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
labels = ['重要保持客户','重要发展客户','重要挽留客户','一般客户','低价格客户']
drawRader(itemnames=itemnames,data=data,title=title,labels=labels, saveas = '2.jpg',rgrids=rgrids)
```
![客户群特征分析图](/example/chapter7/demo/code/radar.jpg)
## 八、[中医证型关联规则挖掘](/example/chapter8/demo)
### 8-1、背景与挖掘目标
- 根据相关数据建模，获取中医证型与乳腺癌之间的关联关系，对治疗提供依据，挖掘潜性证素。
### 8-2、数据探索分析及数据预处理
- 数据预处理：调用k-means算法，进行聚类离散化。
### 8-3、模型构建
- 采用Apriori关联规则算法，挖掘它们之间的关联关系，探索乳腺癌症患者TNM分期与中医证型系数之间的关系。
## 九、[基于水色图像的水质评价](/example/chapter9/demo)
### 9-1、背景与挖掘目标
- 利用图像处理技术，通过水色图像实现水质的自动评价。
### 9-2、数据探索分析及数据预处理
- 数据预处理：图片切割，图片数据特征提取:常用直方图法、颜色矩。
### 9-3、模型构建
- 对特征提取后的样本进行抽样，抽取80%作为训练样式，20%作为测试样本，用于水质评价检验。采用支持向量机作为水质评价分类模型，模型的输入包括两部分，一是训练样本的输入，二是建模参数的输入。
## 十、[家用电器用户行为分析与事件识别](/example/chapter10/demo)
### 10-1、背景与挖掘目标
-  根据热水器厂商提供的数据进行分析，对用户的用水事件进行分析，判断用水是否是洗浴事件，识别不同用户的用水习惯，以提供个性化的服务。(二分类)
### 10-2、数据探索分析及数据预处理
- 数据探索：通过频率分布直方图分析用户用水停顿时间间隔的规律性；然后，探究划分一次完整用水事件的时间间隔阈值。
```
#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas import DataFrame
inputfile = '../data/water_heater.xls' #输入数据路径,需要使用Excel格式
data = pd.read_excel(inputfile,encoding='utf-8')
data[u'发生时间'] = pd.to_datetime(data[u'发生时间'], format='%Y%m%d%H%M%S')  # 将该特征转成日期时间格式（***）
data = data[data[u'水流量'] > 0]  # 只要流量大于0的记录
# print len(data) #7679
data[u'用水停顿时间间隔'] = data[u'发生时间'].diff() / np.timedelta64(1, 'm')  # 将datetime64[ns]转成 以分钟为单位（*****）
data = data.fillna(0)  # 替换掉data[u'用水停顿时间间隔']的第一个空值
#-----第*1*步-----数据探索，查看各数值列的最大最小和空值情况
data_explore = data.describe().T
data_explore['null'] = len(data)-data_explore['count']
explore = data_explore[['min','max','null']]
explore.columns = [u'最小值',u'最大值',u'空值数']
# ----第*2*步-----离散化与面元划分
# 将时间间隔列数据划分为0~0.1，0.1~0.2，0.2~0.3....13以上，由数据描述可知，
# data[u'用水停顿时间间隔']的最大值约为2094，因此取上限2100
Ti = list(data[u'用水停顿时间间隔'])  # 将要面元化的数据转成一维的列表
timegaplist = [0.0, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2100]  # 确定划分区间
cats = pd.cut(Ti, timegaplist, right=False)  # 包扩区间左端,类似"[0,0.1)",（默认为包含区间右端）
x = pd.value_counts(cats)
x.sort_index(inplace=True)
dx = DataFrame(x, columns=['num'])
dx['fn'] = dx['num'] / sum(dx['num'])
dx['cumfn'] = dx['num'].cumsum() / sum(dx['num'])
f1 = lambda x: '%.2f%%' % (x * 100)
dx[['f']] = dx[['fn']].applymap(f1)
# -----第*3*步-----画用水停顿时间间隔频率分布直方图
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
dx['fn'].plot(kind='bar')
plt.ylabel(u'频率/组距')
plt.xlabel(u'时间间隔（分钟）')
p = 1.0 * dx['fn'].cumsum() / dx['fn'].sum()  # 数值等于 dx['cumfn']，但类型是列表
dx['cumfn'].plot(color='r', secondary_y=True, style='-o', linewidth=2)
plt.annotate(format((p[4]), '.4%'), xy=(7, p[4]), xytext=(7 * 0.9, p[4] * 0.95),
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))  # 添加注释，即85%处的标记。这里包括了指定箭头样式。
plt.ylabel(u'累计频率')
plt.title(u'用水停顿时间间隔频率分布直方图')
plt.grid(axis='y', linestyle='--')
# fig.autofmt_xdate() #自动根据标签长度进行旋转
for label in ax.xaxis.get_ticklabels():  # 此语句完成功能同上,但是可以自定义旋转角度
    label.set_rotation(60)
#plt.savefig('../data/Water-pause-times.jpg')
plt.show()
```
![销售类型和销售模式特征分析图](/example/chapter10/demo/data/Water-pause-times.jpg)
### 10-3、模型构建
- 使用Keras库来训练神经网络，训练样本为根据用户记录的日志标记好的用水事件。根据样本，得到训练好的神经网络后，就可以用来识别对应用户的洗浴事件。
### 10-4、模型检验
- 根据用水日志来判断事件是否为洗浴与多层神经网络模型识别结果的比较，检验模型的准确率。
## 十一、[应用系统负载分析与磁盘容量预测](/example/chapter11/demo)
### 11-1、背景与挖掘目标
- 根据历史磁盘数据，采用时间序列分析法，来预测应用系统服务器磁盘已经使用空间的大小；为管理员提供定制化的预警提示。(时间序列---回归)
### 11-2、数据探索分析及数据预处理
- 数据特征分析：通过下图可以发现，磁盘的使用情况都不具有周期性，表现出缓慢性增长，呈现趋势性。可以初步确认数据是非平稳的。
```
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
```
![时序图](/example/chapter11/demo/data/c.jpg)
### 11-3、模型构建
- 模型检验：为了方便对模型进行评价，将经过数据预处理后的建模数据划分两部分，一是建模样本，二是模型验证数据。本例确定的ARIMA（0,1,1）模型通过检验。
### 11-4、模型评价
- 为了评价时序预测模型效果的好坏，选择建模数据的后5条记录作为实际值，将预测值与实际值进行误差分析。误差在可接受范围内，就可对模型进行应用，实现对应用系统容量的预测。
## 十二、[电子商务网站用户行为分析及服务推荐](/example/chapter12/demo)
### 12-1、背景与挖掘目标
- 分析某网站的用户访问记录，然后分析网页相关主题，分析用户网上访问行为；借助用户的访问记录，发现用户的访问习惯，对不同用户进行相关服务页面的推荐。推荐算法
### 12-2、数据探索分析及数据预处理
- 网页类型分析：统计各个网页类型所占的比例；
- 点击次数分析：统计分析原始数据用户浏览网页次数（以“真实IP”区分）的情况
- 网页排名分析：获得各个网页点击率排名以及类型点击率排名：统计分析原始数据用户浏览网页次数（以“真实IP”区分）
### 12-3、模型构建
- 推荐算法：本例所用的算法为协同过滤中的基于物品的过滤，即将哪个物品推荐给某个用户。
### 12-4、模型评价
- 本例采用的是最基本的协同过滤算法进行建模，因些得出的模型结果也是一个初步的效果，实际应用中要结合业务进行分析。
## 十三、[财政收入影响因素分析及预测模型](/example/chapter13/demo)
### 13-1、背景与挖掘目标
- 根据1994-2013年相关财政数据 ，梳理影响地方财政收入的关键特征，对未来几年的财政数据进行预测。 本例用到了回归算法。
### 13-2、数据探索分析及数据预处理
- 探索性分析：对变量进行描述性分析和相关性分析。
### 13-3、模型构建
- 利用了Adaptive-Lasso进行变量选择，AdaptiveLasso算法，要在较新的Scikit-Learn才有，进行预测模型构建。
## 十四、[基于基站定位数据的商圈分析](/example/chapter14/demo)
### 14-1、背景与挖掘目标
- 
### 14-2、数据探索分析及数据预处理
- 
### 14-3、模型构建
- 
## 十五、[电商产品评论数据情感分析](/example/chapter15/demo)
### 15-1、背景与挖掘目标
- 
### 15-2、数据探索分析及数据预处理
- 
### 15-3、模型构建
- 
## 十六、[企业偷漏税识别模型](/example/chapter16/demo)
### 16-1、背景与挖掘目标
- 依据汽车销售企业的部分经营指标的数据，来评估汽车销售行业纳税人的偷漏税倾向，建立偷漏税行为识别模型。
### 16-2、数据探索分析及数据预处理
- 数据分析：其中一个是纳税人编号，一个是是否偷漏税的输出结果，不偷税为正常，存在偷漏税则为异常，其他都为与偷漏税相关的经营指标。本例将分别从分类变量和数值型变量两个方面入手对数据做一个探索性分析<br>
分类变量：销售的汽车类型和销售模式可能会对偷漏税倾向有一定的表征，画出【输出结果为异常的销售类型和销售模式】的分布图可以直观上看出是否有一定影响。<br>
```
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
```
![销售类型和销售模式特征分析图](/example/chapter16/demo/data/plot.jpg)<br>
数值型变量：用describe输出数值型数据分布情况
- 数据预处理：在模型建立时，将分类变量转换成虚拟变量，本例主要对销售类型、销售模式以及输出进行虚拟变量的建立。 
Pandas中有直接转换的函数，get_dummies。
### 16-3、模型构建
- 选取80%作为训练数据，20%作为测试数据。采用CART决策树模型和逻辑回归模型，建模识别企业偷漏税，并绘制了两个模型的ROC曲线进行两个模型的比较。<br>
通过两个模型结果分析：维修毛利、代办保险率、4S店对偷漏税有明显的负相关，成本费用利润率、办牌率、大客车和一级代理商对偷漏税有明显的正相关。<br>
纳税人维修毛利越高，代办保险率越高，通过4S店销售，其偷漏税倾向将会越低；而纳税人成本费用利润率、办牌率越高，销售类型为大客车，销售模式为一级代理商，那么该纳税人将更有可能为偷漏税用户。
### 16-4、模型评价
- 做出的两个模型的ROC曲线如下图所示
```
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
```
![决策树和逻辑回归模型比较ROC图](/example/chapter16/demo/data/dt_lr_roc.jpg)
- ROC曲线越靠近左上角，则模型性能越优，当两个曲线做于同一个坐标时，若一个模型的曲线完全包住另一个模型，则前者优，当两者有交叉时，则看曲线下的面积，上图明显蓝色线下的面积更大，即CART决策树模型性能更优。 
- 结论：对于本例子来说，CART决策树模型不管从混淆矩阵来看，还是从ROC曲线来看，其性能都要优于逻辑回归模型。