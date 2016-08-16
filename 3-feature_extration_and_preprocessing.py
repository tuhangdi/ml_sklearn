# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:50:19 2016

@author: hd
"""

from sklearn.feature_extraction import DictVectorizer

onehot_encoder = DictVectorizer()
# DictVectorizer类可以用来表示分类特征
instances = [{'city': 'New York'}, {'city': 'San Francisco'}, {'city': 'Chapel Hill'}]
print(onehot_encoder.fit_transform(instances).toarray())


#文集包括8个词：UNC, played, Duke, in, basketball, lost, the, game。文件的单词构成词汇表
#（vocabulary）。词库模型用文集的词汇表中每个单词的特征向量表示每个文档。我们的文集有8个
#单词，那么每个文档就是由一个包含8位元素的向量构成。构成特征向量的元素数量称为维度
#（dimension）。用一个词典（dictionary）来表示词汇表与特征向量索引的对应关系。
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game'
]
vectorizer = CountVectorizer()
#CountVectorizer类会把文档全部转换成小写，通过正则表达式用空格分割句子，然后抽取长度大于等于2的字母序列。
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)

corpus = [
'UNC played Duke in basketball',
'Duke lost the basketball game',
'I ate a sandwich'
]
vectorizer = CountVectorizer()
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)
#词汇表里面有10个单词，但a不在词汇表里面，是因为a的长度不符合CountVectorizer类的要求。

from sklearn.metrics.pairwise import euclidean_distances
counts = vectorizer.fit_transform(corpus).todense()
for x,y in [[0,1],[0,2],[1,2]]:
    dist = euclidean_distances(counts[x],counts[y])
    print('文档{}与文档{}的距离{}'.format(x,y,dist))
#两向量的欧式距离就是两个向量的欧式范数或L2范数差的绝对值d = ∥ ∥ x0 − x1 ∥ ∥
#向量的欧式范数就是其元素平方和的平方根    