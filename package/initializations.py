# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:14:34 2017

@author: Elina Thibeau-Sutre
"""

import GMM
import VBGMM
import kmeans

import numpy as np
import random

def initialization_random(n_components,points):
    """
    This method returns an array of k points which will be used in order to
    initialize a k_means algorithm
    
    @param points: an array of points                               (n_points,dim)
    @param k: the number of clusters                                (int)
    @return: an array containing the means of each cluster          (n_components,dim)
    """

    n_points,_ = points.shape
    idx = np.random.randint(n_points,size = n_components)
    
    means = points[idx,:]
    
    km = kmeans.Kmeans(n_components)
    km.means = means
    assignements = km._step_E(points)
    
    return means,assignements

def initialization_plus_plus(n_components,points):
    """
    This method returns an array of k points which will be used in order to
    initialize a k_means algorithm
    
    @param points: an array of points                               (n_points,dim)
    @param k: the number of clusters                                (int)
    @return: an array containing the means of each cluster          (n_components,dim)
    """
    n_points,dim = points.shape
    probability_vector = np.arange(n_points)/n_points #All points have the same probability to be chosen the first time
         
    means = np.zeros((n_components,dim))
    
    for i in range(n_components): 
        total_dst = 0          
        
        #Choice of a new value
        value_chosen = random.uniform(0,1)
        idx_point = 0
        value = 0
        while (value<value_chosen) and (idx_point+1<n_points):
            idx_point +=1
            value = probability_vector[idx_point]
        means[i] = points[idx_point]
        
        #Calculation of distances for each point in order to find the probabilities to choose each point
        if i == 0:
            M = np.linalg.norm(points-means[0],axis=1)
            M = M.reshape((n_points,1))
        else:
            M = kmeans.dist_matrix(points,means[:i+1:])
            
        dst_min = np.amin(M, axis=1)
        dst_min = dst_min**2
        total_dst = np.cumsum(dst_min)
        probability_vector = total_dst/total_dst[-1]
        
    km = kmeans.Kmeans(n_components)
    km.means = means
    assignements = km._step_E(points)

    return means,assignements

def initialization_AF_KMC(n_components,points,m=20):
    """
    A method providing good seedings for kmeans inspired by MCMC
    for more information see http://papers.nips.cc/paper/6478-fast-and-provably-good-seedings-for-k-means
    
    @param points: an array of points                               (n_points,dim)
    @param k: the number of clusters                                (int)
    @return: an array containing the means of each cluster          (n_components,dim)
    """
    n_points,dim = points.shape
    means = np.empty((n_components,dim))
    
    #Preprocessing step
    idx_c = np.random.choice(n_points)
    c = points[idx_c]
    M = np.square(kmeans.dist_matrix(points,c.reshape(1,-1)))
    q = 0.5 * M / np.sum(M) + 0.5 / n_points
    q = q.reshape(n_points)
    
    #Main loop
    means[0] = c
    for i in range(n_components-1):
        # We choose a potential candidate
        x_idx = np.random.choice(n_points,p=q)
        x = points[x_idx]
        dist_x = np.linalg.norm(x-means[:i+1:],axis=1).min()
#        dist_x = kmeans3.dist_matrix(x.reshape(1,-1),means[:i+1:]).min()
        # We have m stM[x_idx]eps to improve this potential new center
        for j in range(m):
            y_idx = np.random.choice(n_points,p=q)
            y = points[y_idx]
            dist_y = np.linalg.norm(y-means[:i+1:],axis=1).min()
            if dist_x*q[y_idx] != 0:
                quotient = dist_y*q[x_idx]/(dist_x*q[y_idx])
            else:
                quotient = 2.0

            if quotient > random.uniform(0,1):
                x_idx = y_idx
                x = y
                dist_x = dist_y
        means[i+1] = x
        
    km = kmeans.Kmeans(n_components)
    km.means = means
    assignements = km._step_E(points)
    
    return means,assignements

def initialization_k_means(n_components,points):
    """
    This method returns an array of k means which will be used in order to
    initialize an EM algorithm
    
    @param points: an array of points                               (n_points,dim)
    @param k: the number of clusters                                (int)
    @return: an array containing the means of each cluster          (n_components,dim)
    """
    km = kmeans.Kmeans(n_components)
    km.fit(points)
    assignements = km.predict_assignements(points)
    
    return km.means,assignements

def initialization_GMM(n_components,points_data,points_test=None,covariance_type="full"):
    """
    This method returns an array of k means and an array of k covariances (dim,dim)
    which will be used in order to initialize an EM algorithm
    
    @param points: an array of points                               (n_points,dim)
    @param k: the number of clusters                                (int)
    @return: an array containing the means of each cluster          (n_components,dim)
             an array containing the covariances of each cluster    (n_components,dim,dim)
    """
    
    GM = GMM.GaussianMixture(n_components,covariance_type=covariance_type)
    GM.fit(points_data,points_test,patience=0)
    log_assignements = GM.predict_log_resp(points_data)
    
    return GM.means,GM.cov,GM.log_weights,log_assignements

def initialization_VBGMM(n_components,points_data,points_test=None,covariance_type="full"):
    """
    This method returns an array of k means and an array of k covariances (dim,dim)
    which will be used in order to initialize an EM algorithm
    
    @param points: an array of points                               (n_points,dim)
    @param k: the number of clusters                                (int)
    @return: an array containing the means of each cluster          (n_components,dim)
             an array containing the covariances of each cluster    (n_components,dim,dim)
    """
    
    GM = VBGMM.VariationalGaussianMixture(n_components)
    GM.fit(points_data,points_test,patience=0)
    log_assignements = GM.predict_log_resp(points_data)
    
    return GM.means,GM.cov,GM.log_weights,log_assignements

def initialize_log_assignements(init,n_components,points_data,points_test=None,covariance_type="full"):
    """
    This method initializes the Variational Gaussian Mixture by setting the values
    of the means, the covariances and the log of the weights.
    
    @param points: an array             (n_points,dim)
    @return: the initial means          (n_components,dim)
             the initial covariances    (n_components,dim,dim)
    """
    
    log_assignements = None
    
    if (init == "random"):
        _,assignements = initialization_random(n_components,points_data)
    elif(init == "plus"):
        _,assignements = initialization_plus_plus(n_components,points_data)
    elif(init == "AF_KMC"):
        _,assignements = initialization_AF_KMC(n_components,points_data)
    elif(init == "kmeans"):
        _,assignements = initialization_k_means(n_components,points_data)
    elif(init == "GMM"):
        _,_,_,log_assignements = initialization_GMM(n_components,points_data,points_test,covariance_type)
    elif(init == "VBGMM"):
        _,_,_,log_assignements = initialization_VBGMM(n_components,points_data,points_test,covariance_type)
        
    if log_assignements is None:
        epsilon = np.finfo(assignements.dtype).eps
        assignements += epsilon
        assignements /= 1 + n_components * epsilon
        log_assignements = np.log(assignements)
    
    return log_assignements

def initialize_mcw(init,n_components,points_data,points_test=None,covariance_type="full"):
    """
    This method initializes the Variational Gaussian Mixture by setting the values
    of the means, the covariances and the log of the weights.
    
    @param points: an array             (n_points,dim)
    @return: the initial means          (n_components,dim)
             the initial covariances    (n_components,dim,dim)
    """
    
    n_points,dim = points_data.shape
    
    log_weights = - np.log(n_components) * np.ones(n_components)
    
    # Warning : the algorithm is very sensitive to these first covariances given
    if covariance_type == "full":
        cov_init = np.cov(points_data.T) / n_components**2
        cov = np.tile(cov_init, (n_components,1,1))
    elif covariance_type == "spherical":
        cov_init = np.var(points_data, axis=0, ddof=1).mean() / n_components**2
        cov = cov_init * np.ones(n_components)
    
    if (init == "random"):
        means,_ = initialization_random(n_components,points_data)
    elif(init == "plus"):
        means,_ = initialization_plus_plus(n_components,points_data)
    elif(init == "AF_KMC"):
        means,_ = initialization_AF_KMC(n_components,points_data)
    elif(init == "kmeans"):
        means,_ = initialization_k_means(n_components,points_data)
    elif(init == "GMM"):
        means,cov,log_weights,_ = initialization_GMM(n_components,points_data,points_test,covariance_type)
    elif(init == "VBGMM"):
        means,cov,log_weights,_ = initialization_VBGMM(n_components,points_data,points_test,covariance_type)
    
    return means,cov,log_weights
    
if __name__ == '__main__':
    
    import pickle
    
    path = 'D:/Mines/Cours/Stages/Stage_ENS/Code/data/data.pickle'
    with open(path, 'rb') as fh:
        data = pickle.load(fh)
    
    k=100
    N=1500
        
    points = data['BUC']
    points_data = points[:N:]
    points_test = points[N:2*N:]
    
    _,assignements1 = initialization_random(k,points_data)
    _,assignements2 = initialization_AF_KMC(k,points_data)
    _,assignements3 = initialization_plus_plus(k,points_data)
    M,C,W = initialize_mcw('VBGMM',k,points_data)