# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 11:51:21 2017

@author: Calixi
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy.random as rnd
import GMM
import CSVreader

if __name__ == '__main__':

    k=4
    points = CSVreader.read("D:/Mines/Cours/Stages/Stage_ENS/Problem/EMGaussienne.data")
    means = CSVreader.initialization_random(points,k)
    points = points[:,0:-1]
    means = means[:,0:-1]
    dim = len(points[0])
    nb_points = len(points)
    cov = np.ones(k)    
    prior_prob = np.ones(k) * 1/float(k)
    
    assignements = GMM.step_E(points,means,cov,prior_prob,full_covariance=False)
    
    #Duplication in order to create matrices k*nb_points*dim
    assignements_duplicated = np.tile(assignements, (dim,1,1)).T
    points_duplicated = np.tile(points, (k,1,1))
    means_duplicated = np.transpose(np.tile(means,(nb_points,1,1)), (1,0,2))
    
    points_centered = points_duplicated - means_duplicated
    points_centered_weighted = np.transpose(assignements_duplicated * points_centered, (0,2,1))
    covariance = np.dot(points_centered,points_centered_weighted)
    covariance = np.transpose(covariance, (0,2,3,1))
    covariance = np.diagonal(covariance)
    covariance = np.diagonal(covariance).T
    covariance = np.sum(covariance, axis=0)
    
    #Duplication in order to create matrices k*dim*dim
    sum_assignement = np.sum(assignements,axis=0)
    sum_assignement = np.reciprocal(sum_assignement)
    
    result = covariance * sum_assignement / dim
    