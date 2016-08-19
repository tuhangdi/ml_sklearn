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
from sklearn.tree import export_graphviz
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
#创建Pipeline和DecisionTreeClassifier类的实例，
# 将criterion参数设置成entropy，这样表示使用信息增益启发式算法建立决策树。

parameters = {
    'clf__max_depth': (150, 155, 160),
    'clf__min_samples_split': (1, 2, 3),
    'clf__min_samples_leaf': (1, 2, 3)
}
#确定网格搜索的参数范围

grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1, scoring='f1')
grid_search.fit(x_train, y_train)
print(u'最佳效果：%0.3f' % grid_search.best_score_)
print(u'最优参数：')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))
predictions = grid_search.predict(x_test)
print(classification_report(y_test, predictions))

"""决策树集成"""
#采用随机森林，把DecisionTreeClassifier替换成RandomForestClassifier。
import pandas as pa
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.tree import export_graphviz

df = pd.read_csv('D:\Users\hd\SourceTree\ml_sklearn/ad.data', header=None, low_memory=False)

explanatory_variable_columns = set(df.columns.values)
response_variable_column = df[len(df.columns.values)-1]
explanatory_variable_columns.remove(len(df.columns.values)-1)

y = [1 if e == 'ad.' else 0 for e in response_variable_column]
x = df.loc[:, list(explanatory_variable_columns)]
x.replace(to_replace=' *\?', value=-1, regex=True, inplace=True)
x_train, x_test, y_train, y_test = train_test_split(x, y)

pipeline = Pipeline([
    ('clf', RandomForestClassifier(criterion='entropy'))
])
parameters = {
    'clf__n_estimators': (5, 10, 20, 50),
    'clf__max_depth': (50, 150, 250),
    'clf__min_samples_split': (1, 2, 3),
    'clf__min_samples_leaf': (1, 2, 3)
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1, scoring='f1')
grid_search.fit(x_train, y_train)
print(u'最佳效果：%0.3f' % grid_search.best_score_)
print(u'最优参数：')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))

predictions = grid_search.predict(x_test)
print(classification_report(y_test, predictions))