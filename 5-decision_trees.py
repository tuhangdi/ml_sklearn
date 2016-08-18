#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2016/8/18 16:30
@Author  : Thd
@Site    : 
@File    : 5-decision_trees.py
@Software: PyCharm
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
#读取数据文件，并将解释变量和相应变量分开
df = pd.read_csv('D:\Users\hd\SourceTree\ml_sklearn/ad.data', header=None, low_memory=False)
explanatory_variable_columns = set(df.columns.values)
response_variable_column = df[len(df.columns.values)-1]
explanatory_variable_columns.remove(len(df.columns.values)-1)
y = [1 if e == 'ad.' else 0 for e in response_variable_column]
x = df.loc[:, list(explanatory_variable_columns)]

x.replace(to_replace=' *\?', value=-1, regex=True, inplace=True)
#超过1/4的图片其宽带或高度的值不完整，数据文件中用空白加问号（“ ?”）表示，我们用正则表达式替换为-1，方便计算。
x_train, x_test, y_train, y_test = train_test_split(x, y)

pipeline = Pipeline([
    ('clf', DecisionTreeClassifier(criterion='entropy'))
])
