#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2016/8/19 17:16
@Author  : Thd
@Site    : 
@File    : test.py
@Software: PyCharm
"""
import numpy as np
x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
print np.array(list(zip(x1, x2))).reshape(len(x1), 2)