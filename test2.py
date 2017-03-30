# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 18:20:12 2017

@author: Calixi
"""

import CSVreader
import GMM
import numpy as np
import math

if __name__ == '__main__':
    
    k=5
    list_points = CSVreader.read("D:/Mines/Cours/Stages/Stage_ENS/Problem/EMGaussienne.data")
    list_means = CSVreader.initialization_plus_plus(list_points,k)
    dim = len(list_points[0])
    nb_points = len(list_points)
    list_cov = np.asarray([np.eye(dim,dim) for i in range(k)])    
    list_prior_prob = np.ones(k) * 1/float(k)
    
    list_soft_assignement = GMM.step_E(list_points,list_means,list_cov,list_prior_prob)
    

    d = np.linspace(0,3000,num=3000)
    A = np.reshape(d, (20,50,3))
    B = np.ones((20,3,3))
#    print(A[0,0,:])
#    print(B[0,:,1])
    C = np.dot(A,B)
    D = np.diagonal(np.transpose(C, (0,2,3,1)))
    
    mat = np.asarray([[1,2],[1,1]])
    mat_rep = np.tile(mat, (4,1,1))
    det = np.linalg.det(mat_rep)

    #Duplication in order to create matrices k*nb_points*dim
    points = np.tile(list_points, (k,1,1))
    means = np.transpose(np.tile(list_means,(nb_points,1,1)), (1,0,2))
    
    points_centered = points - means
    cov_inv = np.linalg.inv(list_cov)
    product = np.dot(points_centered,cov_inv)
    product = np.diagonal(np.transpose(product, (0,2,3,1))).T
                         
    points_centered = np.transpose(points_centered, (0,2,1))
    product = np.dot(product,points_centered)  
    product = np.diagonal(np.transpose(product, (0,2,3,1)))
    product = np.diagonal(product).T
    
                         
    det_covariance = np.linalg.det(list_cov)
    det_covariance = np.reciprocal(det_covariance)
    det_covariance = np.tile(det_covariance, (nb_points,1))
    
    result = (2*math.pi)**(-float(dim)/2) * det_covariance**0.5 * np.exp(-0.5 * product)