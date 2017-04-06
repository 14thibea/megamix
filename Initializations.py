# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:14:34 2017

@author: Calixi
"""

import numpy as np
import random
import kmeans2

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
               
    list_means = np.zeros((k,dimension))
    
    for i in range(k): 
        total_dst = 0          
        
        #Choice of a new seed
        value_chosen = random.uniform(0,1)
        idx_point = 0
        value = 0
        while (value<value_chosen) and (idx_point+1<nb_points):
            idx_point +=1
            value = probability_vector[idx_point]
        list_means[i] = list_points[idx_point]
        
        #Calculation of distances for each point in order to find the probabilities to choose each point
        M = kmeans2.dist_matrix(list_points,list_means,i+1)
        
        for i in range(len(list_points)):
            dst = np.min(M[i])
            total_dst += dst**2
            probability_vector[i] = total_dst
        
        probability_vector = probability_vector/total_dst

    return list_means

def initialization_k_means(points,k):
    """
    This method returns an array of k points which will be used in order to
    initialize a k_means algorithm
    
    @param points: an array of points
    @param k: the number of clusters
    @return: an array of k points which are means of clusters
    """
    means = kmeans2.k_means(points,k)
    return means