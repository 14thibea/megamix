# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:14:34 2017

@author: Calixi
"""

import GMM3
import kmeans3

import numpy as np
import random

def initialization_full_covariances(points,k):
    """
    This method returns the covariances array for methods like kmeans which
    don't deal with covariances
    
    @param points: an array of points                               (n_points,dim)
    @param k: the number of clusters                                (int)
    @return: an array containing the covariances of each cluster    (k,dim,dim)
    """
    variance = points.var(axis=0)
    
    cov = np.diag(variance)
    return cov

def initialization_spherical_covariances(points,k):
    """
    This method returns the covariances array for methods like kmeans which
    don't deal with covariances
    
    @param points: an array of points                               (n_points,dim)
    @param k: the number of clusters                                (int)
    @return: an array containing the covariances of each cluster    (k,dim,dim)
    """
    _,dim= points.shape
    
    minima = points.min(axis=0)
    maxima = points.max(axis=0)
    diff = maxima - minima
    coeff = np.sum(diff)/dim
    coeff = (coeff/(10*k))**2
    
    return coeff

def initialization_random(points,k):
    """
    This method returns an array of k points which will be used in order to
    initialize a k_means algorithm
    
    @param points: an array of points                               (n_points,dim)
    @param k: the number of clusters                                (int)
    @return: an array containing the means of each cluster          (k,dim)
             an array containing the covariances of each cluster    (k,dim,dim)
    """

    n_points = len(points)
    idx = np.random.randint(n_points,size = k)
    return points[idx,:]

def initialization_plus_plus(points,k):
    """
    This method returns an array of k points which will be used in order to
    initialize a k_means algorithm
    
    @param points: an array of points                               (n_points,dim)
    @param k: the number of clusters                                (int)
    @return: an array containing the means of each cluster          (k,dim)
             an array containing the covariances of each cluster    (k,dim,dim)
    """
    n_points,dim = points.shape
    probability_vector = np.arange(n_points)/n_points #All points have the same probability to be chosen the first time
               
    means = np.zeros((k,dim))
    
    for i in range(k): 
        total_dst = 0          
        
        #Choice of a new seed
        value_chosen = random.uniform(0,1)
        idx_point = 0
        value = 0
        while (value<value_chosen) and (idx_point+1<n_points):
            idx_point +=1
            value = probability_vector[idx_point]
        means[i] = points[idx_point]
        
        #Calculation of distances for each point in order to find the probabilities to choose each point
        M = kmeans3.dist_matrix(points,means,i+1)
        
        for i in range(n_points):
            dst = np.min(M[i])
            total_dst += dst**2
            probability_vector[i] = total_dst
        
        probability_vector = probability_vector/total_dst
    

    return means

def initialization_k_means(points,k):
    """
    This method returns an array of k means which will be used in order to
    initialize an EM algorithm
    
    @param points: an array of points                               (n_points,dim)
    @param k: the number of clusters                                (int)
    @return: an array containing the means of each cluster          (k,dim)
             an array containing the covariances of each cluster    (k,dim,dim)
    """
    means,assignements = kmeans3.k_means(points,k)
    
    kmeans3.create_graph(points,means,assignements,"MFCC")
    
    return means

def initialization_GMM(k,points_data,points_test=None):
    """
    This method returns an array of k means and an array of k covariances (dim,dim)
    which will be used in order to initialize an EM algorithm
    
    @param points: an array of points                               (n_points,dim)
    @param k: the number of clusters                                (int)
    @return: an array containing the means of each cluster          (k,dim)
             an array containing the covariances of each cluster    (k,dim,dim)
    """
    
    GMM = GMM3.GaussianMixture(k)
    if points_test is None:
        GMM.predict_log_assignements(points_data,points_data)
    else:
        GMM.predict_log_assignements(points_data,points_test)
    
    
    return GMM.means,GMM.cov