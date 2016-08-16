# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 11:15:19 2016

@author: hd
"""
from __future__ import division
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname = r"c:\windows\fonts\msyh.ttc", size=10)

def runplt():
    plt.figure()
    plt.title(r"匹萨价格和直径数据",fontproperties=font)
    plt.xlabel(r"直径（英寸）",fontproperties=font)
    plt.ylabel(r"价格（美元）",fontproperties=font)
    plt.axis([0,25,0,25])
    plt.grid(True)
    return plt
    
plt = runplt()
x = [[6],[8],[10],[14],[18]]
y = [[7],[9],[13],[17.5],[18]]
plt.plot(x,y,'k.')
plt.show()

#创建并拟合模型
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)
#fit()分析模型参数
print('预测一张12英寸匹萨价格：$%.2f' % model.predict([12])[0])


plt = runplt()
plt.plot(x,y,'k.')
x2 = [[0],[10],[14],[25]]
model = LinearRegression()
model.fit(x,y)
y2 = model.predict(x2)
#predict()通过fit()算出的模型参数构成的模型
plt.plot(x,y,'k.')
plt.plot(x2,y2,'g-')
plt.show()


#带成本函数/损失函数的模型
plt = runplt()
plt.plot(x,y,'k.')
y3 = [14.25,14.25,14.25,14.25]
y4 = y2 * 0.5 + 5
model.fit(x[1:-1], y[1:-1])
y5 = model.predict(x2)
plt.plot(x,y,'k.',label='y')
plt.plot(x2,y2,'g-.',label='y2')
plt.plot(x2,y3,'r-.',label='y3')
plt.plot(x2,y4,'y-.',label='y4')
plt.plot(x2,y5,'o-',label='y5')
plt.legend()
#显示图例
plt.show()


plt = runplt()
plt.plot(x,y,'k.')
x2 = [[0],[10],[14],[25]]
model = LinearRegression()
model.fit(x,y)
y2 = model.predict(x2)
plt.plot(x,y,'k.')
plt.plot(x2,y2,'g-')
#残差预测值
yr = model.predict(x)
for idx,x in enumerate(x):
    plt.plot([x,x],[y[idx],yr[idx]],'r-')
plt.show()


import numpy as np
print('残差平方和：%.2f' % np.mean((model.predict(x) - y) ** 2))



xbar = (6 + 8 + 10 +14 +18) / 5
variance = ((6 - xbar)**2+(8-xbar)**2+(10-xbar)**2+(14-xbar)**2+(18-xbar)**2)/4
print(variance)
#可直接用numpy算方差
print (np.var([6,8,10,14,18], ddof=1))
#计算协方差
print(np.cov([6,8,10,14,18],[7,9,13,17.5,18])[0][1])


x_test = [[8],[9],[11],[16],[12]]
y_test = [[11],[8.5],[15],[18],[11]]
model = LinearRegression()
x = [[6],[8],[10],[14],[18]]
model.fit(x,y)
R2 = model.score(x_test,y_test)
#计算R方
print R2



#多元线性回归
x = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
y = [[7], [9], [13], [17.5], [18]]
model = LinearRegression()
model.fit(x, y)
x_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y_test = [[11], [8.5], [15], [18], [11]]
predictions = model.predict(x_test)
for i, prediction in enumerate(predictions):
    print('Predicted: %s, Target: %s' % (prediction,y_test[i]))
print('R-squared: %.2f' % model.score(x_test,y_test))

#二次回归
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
x_train = [[6],[8],[10],[14],[18]]
y_train = [[7],[9],[13],[17.5],[18]]
x_test = [[6],[8],[11],[16]]
y_test = [[8],[12],[15],[18]]
regressor = LinearRegression()
regressor.fit(x_train,y_train)
xx = np.linspace(0,26,100)
#规定(起点、终点（包含）、返回array的长度)，返回一个两端点间数值平均分布的array。
#[0,...,26.0]
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
#shape[0]读取矩阵第一维度的长度。xx.reshape(数组最内层，数组次内层，。。，数组最外层)
plt = runplt()
plt.plot(x_train,y_train, 'k.')
plt.plot(xx,yy)
quadratic_featurizer = PolynomialFeatures(degree=2)
#sklearn的类 用来产生多项式
x_train_quadratic = quadratic_featurizer.fit_transform(x_train)
x_test_quadratic = quadratic_featurizer.transform(x_test)
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(x_train_quadratic,y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), 'r-')
plt.show()
print(x_train)
print(x_train_quadratic)
print(x_test)
print(x_test_quadratic)
print(r'一元线性回归 r-squared', regressor.score(x_test,y_test))
print(r'二次回归 r-squared', regressor_quadratic.score(x_test_quadratic,y_test))

#二次回归的r方0.8675 三次回归0.8357、七次回归0.4879 ，出现了过拟合，r方降低
plt = runplt()
plt.plot(x_train, y_train, 'k.')
quadratic_featurizer = PolynomialFeatures(degree=2)
x_train_quadratic = quadratic_featurizer.fit_transform(x_train)
x_test_quadratic = quadratic_featurizer.transform(x_test)
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(x_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), 'r-')

cubic_featurizer = PolynomialFeatures(degree=3)
x_train_cubic = cubic_featurizer.fit_transform(x_train)
x_test_cubic = cubic_featurizer.transform(x_test)
regressor_cubic = LinearRegression()
regressor_cubic.fit(x_train_cubic, y_train)
xx_cubic = cubic_featurizer.transform(xx.reshape(xx.shape[0], 1))
plt.plot(xx, regressor_cubic.predict(xx_cubic))

seventh_featurizer = PolynomialFeatures(degree=7)
x_train_seventh = seventh_featurizer.fit_transform(x_train)
x_test_seventh = seventh_featurizer.transform(x_test)
regressor_seventh = LinearRegression()
regressor_seventh.fit(x_train_seventh, y_train)
xx_seventh = seventh_featurizer.transform(xx.reshape(xx.shape[0], 1))
plt.plot(xx, regressor_seventh.predict(xx_seventh))

plt.show()
print(x_train_cubic)
print(x_test_cubic)
print('二次回归 r-squared', regressor_quadratic.score(x_test_quadratic, y_test))
print('三次回归 r-squared', regressor_cubic.score(x_test_cubic, y_test))
print('七次回归 r-squared', regressor_seventh.score(x_test_seventh, y_test
))