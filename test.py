# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:36:06 2017

@author: Calixi
"""

import utils
import Initializations as Init
import GMM2
import GMM3
import kmeans3
import VBGMM

from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import pickle
import os
import scipy.special


from scipy.misc import logsumexp
import numpy as np

if __name__ == '__main__':
    
    k = 6

    points_data = utils.read("D:/Mines/Cours/Stages/Stage_ENS/Code/data/EMGaussienne.data")

    means,covariance = Init.initialization_GMM(points_data,points_data,k)
    d=0
    for i in range(k):
        
        lambda_, v = np.linalg.eig(covariance[i])
        lambda_sqrt = np.sqrt(lambda_)
        
        # We are looking for the extrema for each dimension
        principal_idx = np.argmax(abs(v[d] * lambda_sqrt))
        
        possible_eig_vectors = np.copy(v[d+1])
        possible_eig_vectors[principal_idx] = 0
        
        principal_idx_2 = np.argmax(abs(possible_eig_vectors * lambda_sqrt))
        
        width = lambda_sqrt[principal_idx] * 2
        height = lambda_sqrt[principal_idx_2] * 2
        
                         
        print()
        print(principal_idx)
        print(principal_idx_2)
        
        print()               
        print("width",width)
        print("height", height)
        if principal_idx > principal_idx_2:
            principal_idx = principal_idx_2
            width,height = height,width
        angle = np.rad2deg(np.arccos(v[d,principal_idx]))
        print(angle)
        
        print()               
        print("width",lambda_sqrt[0]*2)
        print("height", lambda_sqrt[1]*2)
        print(np.rad2deg(np.arccos(v[0, 0])))
    
    
    