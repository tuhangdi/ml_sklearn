# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 11:02:44 2016

@author: hd
"""
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname = r"c:\windows\fonts\msyh.ttc", size=10)

import pandas as pd
df = pd.read_csv('D:\Users\hd\SourceTree\ml_sklearn\winequality-red.csv', sep=';')
df.head()
#读取.csv文件生成dataframe

df.describe()
#获得了一堆描述性统计结果

plt.scatter(df['alcohol'],df['quality'])
#散点图
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title(r'酒精度（Alcohol）与品质（Quality）',fontproperties=font)
plt.show()

plt.scatter(df['volatile acidity'], df['quality'])
plt.xlabel('Volatile Acidity')
plt.ylabel('Quality')
plt.title(r'挥发性酸度（volatile acidity）与品质（ Quality）',fontproperties=font)
plt.show()

import pandas as pd
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
df = pd.read_csv('D:\Users\hd\SourceTree\ml_sklearn\winequality-red.csv', sep=';')
x = df[list(df.columns)[:-1]]
y = df['quality']
x_train, x_test, y_train, y_test = train_test_split(x, y)
#train_test_split把数据集分成训练集和测试集,默认25%测试集
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_predictions = regressor.predict(x_test)
print('R-squared:', regressor.score(x_test, y_test))

from sklearn.cross_validation import cross_val_score
regressor = LinearRegression()
scores = cross_val_score(regressor, x, y, cv=5)
print(scores.mean(), scores)
#cross_val_score函数可以实现交叉检验功能。cv参数将数据集分成了5份。每个分区都会轮流作为测试集使用。

plt.scatter(y_test, y_predictions)
plt.xlabel('实际品质',fontproperties=font)
plt.ylabel('预测品质',fontproperties=font)
plt.title('预测品质与实际品质',fontproperties=font)
plt.show()
#和假设一致，预测品质很少和实际品质完全一致。由于绝大多数训练数据都是一般品质的酒，所以这个模型更适合预测一般质量的酒


#SGD随机梯度下降，用波士顿住房数据的13个解释变量来预测房屋价格
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
data = load_boston()
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target)

x_scaler = StandardScaler()
y_scaler = StandardScaler()
#用StandardScaler做归一化处理
x_train = x_scaler.fit_transform(x_train)
y_train = y_scaler.fit_transform(y_train)
x_test = x_scaler.transform(x_test)
y_test = y_scaler.transform(y_test)

regressor = SGDRegressor(loss='squared_loss')
scores = cross_val_score(regressor, x_train, y_train, cv=5)
print('交叉验证R方值：',scores)
print('交叉验证R方均值：',np.mean(scores))
regressor.fit_transform(x_train, y_train)
print('测试集R方值:', regressor.score(x_test, y_test))