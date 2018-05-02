#!/usr/bin/env python
# coding: utf-8
'''
File Name: EE627_HW8.py
Edit Time: 20180501 1819

Content:
    factorize the matrix by spark
    
Version:
    1.0
'''

import pdb # debug module
import numpy as np
from time import gmtime, strftime
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark import SparkContext
import matplotlib.pyplot as plt


def pic_plot():

    a1 = [0.614646370205,\
            0.471935005174,\
            0.291589844507,\
            0.181250135972]
    a1 = [round(item, 3) for item in a1]
    y1 = [5, 10, 20, 30]

    a2 = [0.498905512823,\
            0.342757066443,\
            0.308446064657,\
            0.291364875421,\
            0.284353350228]
    a2 = [round(item, 3) for item in a2]
    y2 = [2, 5, 10, 20, 30]

    a3 = [9.68561093191e-5,\
            0.000646885875786,\
            0.00308496409831,\
            0.0258608874645,\
            0.152809615238,\
            0.29265563126]
    a3 = [round(item, 3) for item in a3]
    y3 = [2000, 5000, 10000, 20000, 50000, 100000]


#    pdb.set_trace()

    plt.figure(1)
    pic = plt.subplot(131)
    plt.plot(y1, a1)
    plt.plot(y1, a1, 'o')
    plt.title('data size 1e5 and numIter 20')
    plt.suptitle('MSE for different scenatios')
    plt.xlabel('rank')
    for i, j in zip(y1,a1):
        pic.annotate(str(j), xy=(i,j), xytext=(10, 10), textcoords='offset points')

    pic = plt.subplot(132)
    plt.plot(y2, a2)
    plt.plot(y2, a2, 'o')
    plt.title('data size 1e5 and rank 20')
    plt.xlabel('numIter')
    for i, j in zip(y2,a2):
        pic.annotate(str(j), xy=(i,j), xytext=(10, 10), textcoords='offset points')
       
    pic = plt.subplot(133)
    plt.plot(y3, a3)
    plt.plot(y3, a3, 'o')
    plt.title('numIter 20 and rank 20')
    plt.xlabel('data size')
    for i, j in zip(y3,a3):
        pic.annotate(str(j), xy=(i,j), xytext=(10, 10), textcoords='offset points')
    plt.show()

def main():

    pic_plot()
    return
    

    sc = SparkContext()
    data = sc.textFile('/home/z/Documents/python/EE627_HW8/re_u.data')

    pdata = sc.parallelize(data.take(100000))
    ratings = pdata.map(lambda l: l.split(','))\
            .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))\

#    pdb.set_trace()

    sc.setCheckpointDir('target') # need to add this!!!

    rank = 20
    numIter = 30
    model = ALS.train(ratings, rank, numIter)   
    testdata = ratings.map(lambda p: (p[0], p[1]))
    predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
    ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    print("Mean Squared Error = " + str(MSE))

if __name__ == '__main__':
    main()
