#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2016/8/19 10:41
@Author  : Thd
@Site    : 
@File    : 6-clustering_with_k-means.py
@Software: PyCharm
"""
#k-means聚类
# 随机取 K（这里 K=2）个中心点。
#然后计算所有点求到这 K 个中心点的距离，假如点 Pi 离中心点 Si 最近，那么 Pi 属于 Si 类。
#接下来，我们要移动中心点到属于他的类的中心。，取该类所有值的平均值为中心点,中心点此处都为手工计算
#然后重复第 2）和第 3）步，直到，中心点没有移动
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)

import numpy as np
x0 = np.array([7, 5, 7, 3, 4, 1, 0, 2, 8, 6, 5, 3])
x1 = np.array([5, 7, 7, 3, 6, 4, 0, 2, 7, 8, 5, 7])
plt.figure()
plt.axis([-1, 9, -1, 9])
plt.grid(True)
plt.plot(x0, x1, 'k.')
plt.show()

#初始时将第一类的中心设置在第5个样本，第二个类的中心设置在第11个样本
c1 = [1, 4, 5, 9, 11]
c2 = list(set(range(12))-set(c1))
x0c1, x1c1 = x0[c1], x1[c1]
x0c2, x1c2 = x0[c2], x1[c2]
plt.figure()
plt.title(u"第一次迭代后聚类结果", fontproperties=font)
plt.axis([-1, 9, -1, 9])
plt.grid(True)
plt.plot(x0c1, x1c1, 'rx')
plt.plot(x0c2, x1c2, 'g.')
plt.plot(4, 6, 'rx', ms=12.0)
plt.plot(5, 5, 'g.', ms=12.0)
plt.show()

#重新计算两个类的中心点，重新计算各个样本和中心点的距离，得出结果：
c1 = [1, 2, 4, 8, 9, 11]
c2 = list(set(range(12)) - set(c1))
x0c1, x1c1 = x0[c1], x1[c1]
x0c2, x1c2 = x0[c2], x1[c2]
plt.figure()
plt.title(u"第二次迭代后聚类结果", fontproperties = font)
plt.axis([-1, 9, -1, 9])
plt.grid(True)
plt.plot(x0c1, x1c1, 'rx')
plt.plot(x0c2, x1c2, 'g.')
plt.plot(3.8, 6.4, 'rx', ms=12.0)
plt.plot(4.57, 4.14, 'g.', ms=12.0)
plt.show()
#新计算的两个中心点

c1 = [0, 1, 2, 4, 8, 9, 10, 11]
c2 = list(set(range(12)) - set(c1))
x0c1, x1c1 = x0[c1], x1[c1]
x0c2, x1c2 = x0[c2], x1[c2]
plt.figure()
plt.title(u'第三次迭代后聚类结果',fontproperties=font)
plt.axis([-1, 9, -1, 9])
plt.grid(True)
plt.plot(x0c1, x1c1, 'rx')
plt.plot(x0c2, x1c2, 'g.')
plt.plot(5.5, 7.0, 'rx', ms=12.0)
plt.plot(2.2, 2.8, 'g.', ms=12.0)
plt.show()
#再重复上面的方法就会发现类的重心不变了，K-Means会在条件满足的时候停止重复聚类过程。
# 通常，条件是前后两次迭代的成本函数值的差达到了限定值，或者是前后两次迭代的重心位置变化达到了限定值。


#肘部法则：如果问题中没有指定k的值，可以通过肘部法则这一技术来估计聚类数量。
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(3.5, 4.5, (2, 10))
x = np.hstack((cluster1, cluster2)).T
#将两个合成一个，T相当于transpose()

k = range(1, 10)
meandistortions = []
for i in k:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(x)
    meandistortions.append(sum(np.min(cdist(x, kmeans.cluster_centers_, 'euclidean'), axis=1)) / x.shape[0])

plt.plot(k, meandistortions, 'bx-')
plt.xlabel('i')
plt.ylabel(u'平均畸变成都Average distortion', fontproperties=font)
plt.title(u'用肘部法则来确定最佳的i值，Selecting i with the Elobw Method', fontproperties=font)
plt.show()


#聚类效果评估：采用轮廓系数s=(b*a)/max(a,b),a是每一个类中样本彼此距离的均值， 是一个类中样本与其最近的那个类的所有样本的距离的均值
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics

plt.figure(figsize=(8, 10))
plt.subplot(3, 2, 1)
x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
x = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
#zip():接受一系列可迭代对象作为参数，将对象中对应的元素打包成一个个 tuple（元组），然后返回由这些 tuples 组成的 list（列表）。若传入参数的长度不等，则返回 list 的长度和参数中长度最短的对象相同。
#x =[[1 1],[2 3],[3 2],[1 2],[5 8],[6 6],[5 7],[5 6],[6 7],[7 1],[8 2],[9 1],[7 1],[9 3]]
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title(u'样本', fontproperties=font)
plt.scatter(x1, x2)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']
tests = [2, 3, 4, 5, 8]
subplot_counter = 1
for t in tests:
    subplot_counter += 1
    plt.subplot(3, 2, subplot_counter)
    kmeans_model = KMeans(n_clusters=t).fit(x)
    for i, l in enumerate(kmeans_model.labels_):
        plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l], ls='None')
        plt.xlim([0, 10])
        plt.ylim([0, 10])
        plt.title(u'k = %s, 轮廓系数 = %.03f' % (t, metrics.silhouette_score(x, kmeans_model.labels_, metric='euclidean')), fontproperties=font)
plt.show()