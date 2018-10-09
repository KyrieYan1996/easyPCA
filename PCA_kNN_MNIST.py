#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 19:36:35 2018

@author: yankeli
"""

import numpy as np
import operator
import time
import matplotlib.pyplot as plt

#def my_PCA(Mat,n_components):
#    global train_mean
#    meanRemoved = Mat-train_mean
#    covMat = np.cov(meanRemoved,rowvar=0)
#    eigVals,eigVecs = np.linalg.eig(covMat)
#    idx = np.argsort(eigVals)[::-1][:n_components]
#    reserveVec = eigVecs[:,idx]
#    lowDataMat = np.matmul(meanRemoved,reserveVec)
#    return lowDataMat,reserveVec


def kNN(predict_data,train_data,train_label,k):
    datasetSize = train_data.shape[0]
    diff = np.tile(predict_data,(datasetSize,1))-train_data
    square = diff**2
    sqDistance = square.sum(axis = 1)
    distance = np.sqrt(sqDistance)
    sortedDistance = np.argsort(distance)
    classCount = {}
    for i in range(k):
        Vlabel = train_label[sortedDistance[i]]
        classCount[Vlabel] = classCount.get(Vlabel,0)+1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
#*****************train_data***********************#
path = './train.txt'
f = np.loadtxt(path)
label = []
data = []
for i in range(f.shape[0]):
    label.append(np.int(f[i][0]))
    data.append(f[i][1:])
    
data = np.array(data)
train_mean = data.mean(axis = 0)
#****************test_data************************#
path1 = './test.txt'
f1 = np.loadtxt(path1)
label1 = []
data1 = []
for i in range(f1.shape[0]):
    label1.append(np.int(f1[i][0]))
    data1.append(f1[i][1:])
    
data = np.array(data)
data1 = np.array(data1)
l = []

for n_components in range(5,256,5):
    start = time.time()
    #********************PCA*********************#
    meanRemoved = data-train_mean
    covMat = np.cov(meanRemoved,rowvar=0)
    eigVals,eigVecs = np.linalg.eig(covMat)
    idx = np.argsort(eigVals)[::-1][:n_components]
    reserveVec = eigVecs[:,idx]
    lowDataMat = np.matmul(meanRemoved,reserveVec)
    #********************************************#
    yhat = []
    
    test_Mat = data1-train_mean
    test_ = np.matmul(test_Mat,reserveVec)
    
    for idx in range(data1.shape[0]):
        pred = kNN(test_[idx],lowDataMat,label,5)
        yhat.append(pred)
    total = data1.shape[0]
    acc = sum(np.equal(yhat,label1)) / total 
    l.append(acc)
    stop = time.time()
    print('n_components is %d,acc is %.5f,run time is %.5f s'%(n_components,acc,stop-start))
plt.plot(range(5,256,5),l)

