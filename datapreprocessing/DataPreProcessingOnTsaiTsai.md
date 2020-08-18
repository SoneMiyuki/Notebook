# 数据预处理

可以先把数据读成二维列表的形式然后用pd.DataFrame(data),也可以用pd的csv相关东西直接读.
注意针对特征的预处理一定要记得从一维升上去
数据预处理需要在sklearn.preprocessing 中对相关的内容进行import

## Scaler

### 线性变换归一化(normalization):
- 不涉及梯度距离,需要压缩数据
- 实例化MinMaxScaler类

以线性投影的形式把这一个数据项变为[0,1]

$$
 for \qquad x \in S \\
 x' = \frac {x-min(S)} {max(S)-min(S)}
$$

```Python3
scaler = MinMaxScaler()
scaler = scaler.fit(data)
result = scaler.transform(data)
result
```
 - 参数feature_range可以用来调节间距,默认是[0,1].

```Python3
scaler = MinMaxScaler(feature_range = [5,10])
result = scaler.fit_transform(data)
###这个方法一行之后可以进行训练与导出
result
```
逆转还原数据
```
scaler.inverse_transform(result)
```

 - fit因数据量太大报错就改成partial_fitI()使用方法类似

numpy实现:
```Python3
X = np.array([],[],[])

X_nor  =(X - X.min(axis = 0))/ (X.max(axis = 0) - X.min(axis = 0))
```

### 标准化(Standardization)
- 常用
- 构造枢纽变量,把数据调整成服从$\mu = 0, \sigma = 1$的分布的变量

$$
    x' = \frac{x-\mu}{\sigma}
$$

```Python3
scaler = StandardScaler()
scaler.fit(data)

scaler.mean_
scaler.var_

x_std = scaler.transform(data)

x_std.mean()
x_std.std()
```

### 最大绝对值
$$
 for \qquad x \in S \\
 x' = \frac {|x|} {max(S)}
$$

适用于点比较稀疏的情况
MaxAbsScaler被用于绝对值缩放,通过特征里的数据除以绝对值最大的数值的绝对值压缩数据到[-1,1]之间,好处是不会破坏数据中0的个数这样的信息,这样的信息被称为稀疏性.

### 无量纲化

RoubustScaler被用于无量纲化,用来中心化.
适用于有一些离群点使得均值方差偏大的情况

Normalizer将样本独立缩放到单位范数,一般用于密集数组或者稀疏矩阵

PowerTranformer使得数据接近正态分布
但注意输入数据要恒为正

QuantileTranformer使用百分数转换特征,缩小边缘异常值和非异常值的距离

KernelCenterer:将X用和函数映射到希尔伯特空间然后进行中心化

## 缺失值

### sklearn.impute

注意到可以用pd.read_csv(path)直接在pandas进行处理,其中index_col是说ID那一列,可以省略展示

探索数据data.info()

#### SimpleImputer()
参数:

- missing_values默认空值np.nan
- strategy填补的策略
- mean均值填补
- median中位数填补
- most_frequent众数填补
- constant用另一个参数fill_value的取值填补
- fill_value:strategy = constant的时候填补值
- copy默认True,创建一个特征矩阵的副本

先取出数据的某一列,然后对其进行修改

```Python3
Age = data.loc[:,'age'].values.reshape(-1,1)

imp.mean = SimpleImputer()
imp_mean  =imp_mean.fit_tranform(Age)
```

或者用pandas
```
data_.loc[:,'Age']=data_.loc[:,'Age].fillna(data_.loc[:,'Age'].median())
data_.dropna(axis = 0,inplace = True)
#axis = 0删行 inplace:是否在原数据集修改
```

## 编码

### LabelEncoder
preprocessing.LabelEncoder

```python3
from sklearn.preprocessing import LabelEncoder
y = data.iloc[:,-1]
#因为这里输入标签,所以只要一维就够了
```
过程
```
le = LabelEncoder()
le = le.fit(y)
label = le.tranform(y)
le.classes_
#可以发现这里可以搞定多分类问题
#这里fit_transform()和inverse_tranform(label)依然可以使用
```
简要写法
```
from sklearn.preprocessing import LabelEncoder
data.iloc[:,-1] = LabelEncoder().fit_tranform(data.iloc[:,-1])
```

### OrdinalEncoder

```Python3
data_ = data.copy()
data_.head()
OrdinalEncoder().fit(data_.iloc[:,1:-1]).categories_#所有行从索引为1的列取到索引为-2的列
data_iloc[:,1:-1]=OrdinalEncoder().fit_tranform(data_.iloc[:,1:-1])
data_.head()
```

### OneHotEncoder

变量类型
- 名义变量(区别不同)
- 序数变量(区别偏序)
- 有距变量(距离意义)
- 比率变量(比也有意)

名义变量可以用OneHotEncoder

```python3
enc  =OneHotEncoder(categories = 'auto').fit(X)
result = enc.transform(X).toarray()
enc.get_feature_names()
```
一步到位
```python3
OnHotEncoder(categories  ='auto).fit_transform(X).toarray()
```

OneHot之后的合并
```python3
newdata = pd.concat([data,pd,DataFrame(result)],axis =1)
newdata.drop(['Sex','Embarked'],axis=1,inplace = True)
newdata.columns = ['','',...]
```

## 二值化和分段

### Binarizer
```python3
data_2 = data.copy()
from sklearn.preprocessing import Binarizer
X = data_2.iloc[:,0].values.reshape(-1,1)
tranformer = Binarizer(threshold = 30).fit_tranform(X)
```

###KBinsDiscretizer

 - n_bins :特征中分箱的个数,默认5
 - encode:编码方式,默认onehot
  - onehot:哑变量,
  - ordinal:每个箱被编码成一个整数
  - onehot-dense:哑变量,返回密集数组
 - strategy:定义箱宽的方式,默认quantile
  - uniform:等宽分箱
  - quantile:等位分箱
  - kmeans:聚类分箱
 
```python3
est = KBinsDiscretizer(n_bins = 3,encoder = 'ordinal',strategy = 'uniform')
est.fit_transform(X)
```




---

参考内容:
[【机器学习】菜菜的sklearn课堂【全85集】Python进阶](https://www.bilibili.com/video/BV1vJ41187hk?p=18)

侵权请联系删除




