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
4、贝叶斯网络：贝叶斯网络又称信度网络，是Byes方法的扩展，是目前不确定知识表达和推理领域最有效的理论模型之一。<br>
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
