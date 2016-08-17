# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 16:09:06 2016

@author: hd
"""
"""
逻辑回归分类垃圾短信
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)

df = pd.read_csv('D:\Users\hd\SourceTree\ml_sklearn\SMSSpamCollection', delimiter='\t', header=None)
x_train_raw, x_test_raw, y_train, y_test = train_test_split(df[1], df[0])
#从样本中随机的按比例选取 train data 和 test data
#X_train, X_test, y_train, y_test
# = cross_validation.train_test_split(train_data, train_target, test_size=0.4, random_state=0)

#建一个TfidfVectorizer实例来计算TF-IDF权重
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train_raw)
x_test = vectorizer.transform(x_test_raw)

#建一个LogisticRegression实例来训练模型
classifier = LogisticRegression()
classifier.fit(x_train, y_train)
predictions = classifier.predict(x_test)

for i, predictions in enumerate(predictions[-5:]):
    print('预测类型：%s. 信息：%s' % (predictions, x_test_raw.iloc[i]))

"""
二元分类效果评估
"""
from sklearn.metrics import confusion_matrix
#混淆矩阵（Confusion matrix），也称列联表分析（Contingencytable）:行表示实际类型，列表示预测类型。
import matplotlib.pyplot as plt
y_test = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
plt.matshow(confusion_matrix)
plt.title(u'混淆矩阵', fontproperties=font)
plt.colorbar()
plt.ylabel(u'实际类型', fontproperties=font)
plt.xlabel(u'预测类型', fontproperties=font)
plt.show()

#准确率：scikit-learn提供了accuracy_score来计算
from sklearn.metrics import accuracy_score
y_pred, y_true = [0, 1, 1, 0], [1, 1, 1, 1]
print(accuracy_score(y_true, y_pred))

#LogisticRegression.score()用来计算模型预测的准确率：
import numpy as np
import  pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score

df = pd.read_csv('D:\Users\hd\SourceTree\ml_sklearn\sms.csv')
x_train_raw, x_test_raw, y_train, y_test = train_test_split(df['message'], df['label'])
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train_raw)
x_test = vectorizer.transform(x_test_raw)
classifier = LogisticRegression()
classifier.fit(x_train, y_train)
scores = cross_val_score(classifier, x_train, y_train, cv=5)
print(u'准确率：',np.mean(scores),scores)
#准确率是分类器预测正确性的比例，但不能分辨出假阳性错误和假阴性错误

#scikit-learn结合真实类型数据，提供了一个函数来计算一组预测值的精确率和召回率。
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score

df = pd.read_csv('D:\Users\hd\SourceTree\ml_sklearn\sms.csv')
x_train_raw, x_test_raw, y_train, y_test = train_test_split(df['message'], df['label'])
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train_raw)
x_test = vectorizer.transform(x_test_raw)
classifier = LogisticRegression()
classifier.fit(x_train, y_train)
precisions = cross_val_score(classifier, x_train, y_train, cv=5, scoring='precision')
print('精确率：', np.mean(precisions), precisions)
recalls = cross_val_score(classifier, x_train, y_train, cv=5, scoring='recall')
print('召回率：', np.mean(recalls), recalls)
#我们的分类器精确率99.2%，分类器预测出的垃圾短信中99.2%都是真的垃圾短信。
# 召回率比较低67.2%，就是说真实的垃圾短信中，32.8%被当作正常短信了，没有被识别出来。
