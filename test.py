# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:36:06 2017

@author: Calixi
"""

import numpy as np
import CSVreader
import kmeans2

if __name__ == '__main__':
    
    k=5
    list_points = CSVreader.read("D:/Mines/Cours/Stages/Stage_ENS/Problem/EMGaussienne.data")
    list_means = kmeans2.initialization_plus_plus(list_points,k)
    M = kmeans2.dist_matrix(list_points,list_means,k)
    print(M)
    distortion = np.sum(M)
    print(distortion)
    M1 = np.asarray(M)
    print(M1*M1)
    