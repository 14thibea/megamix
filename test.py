# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:36:06 2017

@author: Calixi
"""

import CSVreader
import Initializations as Init
import numpy as np
import scipy.stats
import scipy.special

if __name__ == '__main__':
    
    k = 5
    dim = 2
    i = 1
    N = 100
    coeff = 1
    
    points = CSVreader.read("D:/Mines/Cours/Stages/Stage_ENS/Code/Problem/EMGaussienne.data")
    nb_points,dim = points.shape
    
    means = Init.initialization_plus_plus(points,k)
    cov = np.tile(np.eye(dim,dim), (k,1,1))
    a = scipy.special.psi(-0.5)
    print(a)