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

#综合评价指标（F1 measure）:精确率和召回率的调和均值（harmonic mean），或加权平均值.F1=2*((P*R)/(P+R))
fls = cross_val_score(classifier, x_train, y_train, cv=5, scoring='f1')
print('综合评价指标：', np.mean(fls), fls)

#ROC曲线对分类比例不平衡的数据集不敏感，ROC曲线显示的是对超过限定阈值的所有预测结果的分类器效果。
#AUC是ROC曲线下方的面积，它把ROC曲线变成一个值，表示分类器随机预测的效果。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc

df = pd.read_csv('D:\Users\hd\SourceTree\ml_sklearn\sms.csv')
x_train_raw, x_test_raw, y_train, y_test = train_test_split(df['message'], df['label'])
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train_raw)
x_test = vectorizer.transform(x_test_raw)
classifier = LogisticRegression()
classifier.fit(x_train, y_train)
predictions = classifier.predict_proba(x_test)
false_positive_rate, recall, thresholds = roc_curve(y_test, predictions[:, 1])
#假阳性率，召回率，阈值
roc_auc = auc(false_positive_rate, recall)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
#ROC曲线画的是分类器的召回率与误警率（fall-out）(也称假阳性率)的曲线
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.00, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()


#网格搜索：是确定最优超参数的方法，采用穷举法，选取可能的参数不断运行模型获取最佳效果。
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

pipeline = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression())
])
parameters = {
    'vect__max_df': (0.25, 0.5, 0.75),
    'vect__stop_words': ('english', None),
    'vect__max_features': (2500, 5000, 10000, None),
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__use_idf': (True, False),
    'vect__norm': ('l1', 'l2'),
    'clf__penalty':('l1', 'l2'),
    'clf__C':(0.01, 0.1, 1, 10),
}
grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1, scoring='accuracy', cv=3)
#GridSearchCV()函数的参数有待评估模型pipeline，超参数词典parameters和效果评价指标scoring。n_jobs是指并发进程最大数量，设置为-1表示使用所有CPU核心进程(会报错原因未知)。
df = pd.read_csv('D:\Users\hd\SourceTree\ml_sklearn\sms.csv')
x, y, = df['message'], df['label']
x_train, x_test, y_train, y_test = train_test_split(x, y)
grid_search.fit(x_train, y_train)
print(u'最佳效果：%0.3f' % grid_search.best_score_)
print(u'最优参数组合：')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))
predictions = grid_search.predict(x_test)
print(u'准确率：', accuracy_score(y_test, predictions))
print(u'精确率：', precision_score(y_test, predictions))
print(u'召回率：', recall_score(y_test, predictions))
"""


"""
多类分类
"""
import zipfile
df = pd.read_csv('D:\Users\hd\SourceTree\ml_sklearn/train.tsv', header=0, delimiter='\t')

print df.head()
#取前n行，默认维5
print df.count()
#计算每列行数
print df.Phrase.head(10)
print df.Sentiment.describe()
# count    156060.000000
# mean          2.063578
# std           0.893832
# min           0.000000
# 25%           2.000000
# 50%           2.000000
# 75%           3.000000
# max           4.000000
# Name: Sentiment, dtype: float64
print df.Sentiment.value_counts()
# 2    79582
# 3    32927
# 1    27273
# 4     9206
# 0     7072
# Name: Sentiment, dtype: int64
print df.Sentiment.value_counts()/df.Sentiment.count()

#网格搜索获得最佳参数组合
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

pipeline = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression())
])
parameters = {
    'vect__max_df': (0.25, 0.5),
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__use_idf': (True, False),
    'clf__C':(0.1, 1, 10),
}
df = pd.read_csv('D:\Users\hd\SourceTree\ml_sklearn/train.tsv', header=0, delimiter='\t')
x, y = df['Phrase'], df['Sentiment'].as_matrix()
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.5)
grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1, scoring='accuracy')
grid_search.fit(x_train, y_train)
print('最佳效果： %0.3f' % grid_search.best_score_)
print('最优参数组合：')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s: %r'% (param_name, best_parameters[param_name]))

#使用最佳参数组合的分类器
predictions = grid_search.predict(x_test)
print('准确率：',accuracy_score(y_test, predictions))
print('混淆矩阵：', confusion_matrix(y_test,predictions))
print('分类报告：'， classification_report(y_test, predictions))


"""多标签分类"""
#讲每个标签都用二元分类处理。每个标签的分类器都预测样本是否属于该标签。

#多标签分类效果评估：
#汉明损失函数（Hammingloss）:错误标签的平均比例，是一个函数，当预测全部正确，即没有错误标签时，值为0。
#杰卡德相似度（Jaccard similarity）:是预测标签和真实标签的交集数量除以预测标签和真实标签的并集数量。其值在{0,1}之间
import numpy as np
from sklearn.metrics import hamming_loss, jaccard_similarity_score
print(hamming_loss(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[0.0, 0], [1.0, 1.0]])))
print(hamming_loss(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[1.0, 1.0], [1.0, 1.0]])))
print(hamming_loss(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[1.0, 1.0], [0.0, 1.0]])))
print(jaccard_similarity_score(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[0.0, 1.0], [1.0, 1.0]])))
print(jaccard_similarity_score(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[1.0, 1.0], [1.0, 1.0]])))
print(jaccard_similarity_score(np.array([[.0, 1.0], [1.0, 1.0]]), np.array([[1.0, 1.0], [0.0, 1.0]])))