# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 22:02:09 2017

@author: Calixi
"""

import pandas as pd
import numpy as np
import random

def read(file_name):
    fichier = pd.read_csv(file_name,sep = " ")
    matrix = fichier.as_matrix()
    nb_points = len(matrix)
    dimension = len(matrix[0])
    
    list_points = np.zeros((nb_points,dimension+1))
    list_points[:,:-1] = matrix
    return list_points

def initialization_random(list_points,k):
    """
    This method returns an array of k points which will be used in order to
    initialize a k_means algorithm
    
    @param list_points: an array of points
    @param k: the number of clusters
    @return: an array of k points which are means of clusters
    """

    nb_points = len(list_points)
    idx = np.random.randint(nb_points,size = k)
    return list_points[idx,:]

def dist_matrix(list_points,list_means,k):
    """
    This method computes all the distances between the points and the actual means
    of the clusters, taking in account that the last coordinate is the cluster number
    
    @param list_points: an array of points
    @param list_means: an array of k points which are the means of the clusters
    @return: a matrix which contains the distances between the ith point and the jth center
    """
    
    nb_points = len(list_points)
    
    real_points = list_points[:,0:-1]
    square_points_distances = np.sum(real_points*real_points, axis=1)
    A = np.transpose(np.asmatrix([square_points_distances for i in range(k)]))
    # A is a nb_points * k matrix. Aij is the squared norm of the ith point of list_points
    
    real_means = list_means[:,0:-1]
    real_means = real_means[0:k:1]
    real_means = np.asarray(real_means)
    square_means_distances = np.sum(real_means*real_means, axis=1)
    B = np.asmatrix([square_means_distances for i in range(nb_points)])
    # B is a nb_points * k matrix. Bij is the squared norm of the jth point of list_means   
    
    C = np.dot(real_points,np.transpose(real_means))
    # C is a nb_points * k matrix. Cij is the scalar product of the ith point of list_points and the jth point of list_means
    
    return np.sqrt(A+B-2*C)

def initialization_plus_plus(list_points,k):
    """
    This method returns an array of k points which will be used in order to
    initialize a k_means algorithm
    
    @param list_points: an array of points
    @param k: the number of clusters
    @return: an array of k points which are means of clusters
    """
    nb_points = len(list_points)
    dimension = len(list_points[0])
    probability_vector = np.arange(nb_points)/nb_points #All points have the same probability to be chosen the first time
    list_points[:,-1] = probability_vector
               
    list_means = np.zeros((k,dimension))
    
    for i in range(k): 
        total_dst = 0          
        
        #Choice of a new seed
        value_chosen = random.uniform(0,1)
        idx_point = 0
        value = 0
        while (value<value_chosen) and (idx_point+1<nb_points):
            idx_point +=1
            value = list_points[idx_point][-1]
        list_means[i] = list_points[idx_point]
        
        #Calculation of distances for each point in order to find the probabilities to choose each point
        M = dist_matrix(list_points,list_means,i+1)
        
        for point in list_points:
            dst = np.min(M[i])
            total_dst += dst**2
            point[-1] = total_dst
        
        list_points[:,-1] = list_points[:,-1]/total_dst

    return list_means
