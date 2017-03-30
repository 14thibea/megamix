# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:36:06 2017

@author: Calixi
"""

import CSVreader
import kmeans2
import numpy as np

if __name__ == '__main__':
    
    k=5
    list_points = CSVreader.read("D:/Mines/Cours/Stages/Stage_ENS/Problem/EMGaussienne.data")
    list_means = CSVreader.initialization_plus_plus(list_points,k)
    
#    nb_points = len(list_points)
#    mean = list_means[0]
#    array = np.tile(mean, (nb_points, 1))
#    print(array)
#    array_reciproq = np.reciprocal(array)
#    print(array_reciproq)
#    print(list_means)
#    test = np.tile(list_means,(nb_points,1,1))
#    test2 = np.transpose(test,(1,0,2))
#    print(test2)
#    print(list_means[k-1])
#    array = np.asarray([[[0,1,2],[1,2,0]],[[1,1,3],[1,1,3]]])
#    result = np.reshape(array,(2,6))
#    print(result[0])

d = np.linspace(0,2000,num=2000)
A = np.reshape(d, (5,20,20))
B = np.diagonal(A).T