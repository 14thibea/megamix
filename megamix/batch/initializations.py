# -*- coding: utf-8 -*-
#
#Created on Mon Apr  3 15:14:34 2017
#
#author: Elina Thibeau-Sutre
#

import numpy as np
import random


def initialization_random(n_components,points):
    """
    This method returns an array of k points which will be used in order to
    initialize a k_means algorithm
    
    Parameters
    ----------
    points : an array (n_points,dim)
    
    k : int
        the number of clusters
        
    Returns
    -------
    means : an array (n_components,dim)
        The initial means computed
    
    assignements : an array (n_points,n_components)
        The hard assignements according to kmeans
        
    """    
    n_points,_ = points.shape
    idx = np.random.randint(n_points,size = n_components)
    
    means = points[idx,:]
    
    return means

def initialization_random_sklearn(n_components,points):
    """
    This method returns an array of k points which will be used in order to
    initialize a k_means algorithm
    
    Parameters
    ----------
    points : an array (n_points,dim)
    
    k : int
        the number of clusters
        
    Returns
    -------
    means : an array (n_components,dim)
        The initial means computed
    
    assignements : an array (n_points,n_components)
        The hard assignements according to kmeans
        
    """
    n_points,_ = points.shape
    random_state = np.random.RandomState(2)
    resp = random_state.rand(n_points,n_components)
    resp /= resp.sum(axis=1)[:,np.newaxis]
    
    return resp

def initialization_plus_plus(n_components,points,info=False):
    """
    This method returns an array of k points which will be used in order to
    initialize a k_means algorithm
    
    Parameters
    ----------
    points : an array (n_points,dim)
    
    k : int
        the number of clusters
        
    Returns
    -------
    means : an array (n_components,dim)
        The initial means computed
    
    assignements : an array (n_points,n_components)
        The hard assignements according to kmeans
        
    """
    from .kmeans import dist_matrix
    
    dist = None
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
            M = dist_matrix(points,means[:i+1:])
            
        dst_min = np.amin(M, axis=1)
        dst_min = dst_min**2
        total_dst = np.cumsum(dst_min)
        probability_vector = total_dst/total_dst[-1]
        
    if info:
        from .kmeans import Kmeans
        km = Kmeans(n_components)
        km.means = means
        km._is_initialized = True
        dist = km.score(points)
        
        return means, dist

    return means

def initialization_AF_KMC(n_components,points,m=20):
    """
    A method providing good seedings for kmeans inspired by MCMC
    for more information see http://papers.nips.cc/paper/6478-fast-and-provably-good-seedings-for-k-means
    
    Parameters
    ----------
    points : an array (n_points,dim)
    
    k : int
        the number of clusters
        
    Returns
    -------
    means : an array (n_components,dim)
        The initial means computed
    
    assignements : an array (n_points,n_components)
        The hard assignements according to kmeans
        
    """
    from .kmeans import dist_matrix
    
    n_points,dim = points.shape
    means = np.empty((n_components,dim))
    
    #Preprocessing step
    idx_c = np.random.choice(n_points)
    c = points[idx_c]
    M = np.square(dist_matrix(points,c.reshape(1,-1)))
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
    
    return means

def initialization_k_means(n_components,points,info=False):
    """
    This method returns an array of k means which will be used in order to
    initialize an EM algorithm
    
    Parameters
    ----------
    points : an array (n_points,dim)
    
    k : int
        the number of clusters
        
    Returns
    -------
    means : an array (n_components,dim)
        The initial means computed
    
    assignements : an array (n_points,n_components)
        The hard assignements according to kmeans
        
    """
    from .kmeans import Kmeans
    
    km = Kmeans(n_components)
    km.fit(points)
    assignements = km.predict_assignements(points)
    
    if info:
        dist=km.score(points)
        return km.means,assignements,dist

    return km.means,assignements

def initialization_GMM(n_components,points_data,points_test=None,covariance_type="full"):
    """
    This method returns an array of k means and an array of k covariances (dim,dim)
    which will be used in order to initialize an EM algorithm
    
    Parameters
    ----------
    points : an array (n_points,dim)
    
    k : int
        the number of clusters
        
    Returns
    -------
    means : an array (n_components,dim)
        The initial means computed
    
    cov : an array (n_components,dim,dim)
        The initial covariances computed
        
    log_weights : an array (n_components,)
        The initial weights (log) computed
        
    log_assignements : an array (n_points,n_components)
        The log of the soft assignements according to GMM
        
    """
    from .GMM import GaussianMixture
    
    GM = GaussianMixture(n_components,covariance_type=covariance_type)
    GM.fit(points_data,points_test,patience=0)
    log_assignements = GM.predict_log_resp(points_data)
    
    return GM.means,GM.cov,GM.log_weights,log_assignements

def initialization_VBGMM(n_components,points_data,points_test=None,covariance_type="full"):
    """
    This method returns an array of k means and an array of k covariances (dim,dim)
    which will be used in order to initialize an EM algorithm
    
    Parameters
    ----------
    points : an array (n_points,dim)
    
    k : int
        the number of clusters
        
    Returns
    -------
    means : an array (n_components,dim)
        The initial means computed
    
    cov : an array (n_components,dim,dim)
        The initial covariances computed
        
    log_weights : an array (n_components,)
        The initial weights (log) computed
        
    log_assignements : an array (n_points,n_components)
        The log of the soft assignements according to VBGMM
        
    """
    from .VBGMM import VariationalGaussianMixture
    
    GM = VariationalGaussianMixture(n_components)
    GM.fit(points_data,points_test,patience=0)
    log_assignements = GM.predict_log_resp(points_data)
    
    return GM.means,GM.cov,GM.log_weights,log_assignements

def initialize_log_assignements(init,n_components,points_data,points_test=None,covariance_type="full"):
    """
    This method initializes the Variational Gaussian Mixture by giving the value
    of the responsibilities to the algorithm.
    
    Parameters
    ----------
    init : str
        The method with which the algorithm can be initialized.
        Must be in ['random','plus','AF_KMC','kmeans','GMM','VBGMM']
    
    n_components : int
        the number of clusters
        
    points_data : an array (n_points,dim)
    
    covariance_type : str
        Type of covariance : 'full' or 'spherical'
    
    Other Parameters
    ----------------
    points_test : array (n_points_bis,dim) | Optional
        Initializes using early stopping in order to avoid over fitting.
    
    Returns
    -------
    log_assignements : an array (n_points,n_components)
        The log of the soft assignements according to VBGMM
        
    """
    from .kmeans import Kmeans
    log_assignements = None
    
    if (init == "random"):
        means = initialization_random(n_components,points_data)
        km = Kmeans(n_components)
        km.means = means
        assignements = km._step_E(points_data)
    elif(init == "random_sk"):
        assignements = initialization_random_sklearn(n_components,points_data)
    elif(init == "plus"):
        means = initialization_plus_plus(n_components,points_data)
        km = Kmeans(n_components)
        km.means = means
        assignements = km._step_E(points_data)
    elif(init == "AF_KMC"):
        means = initialization_AF_KMC(n_components,points_data)
        km = Kmeans(n_components)
        km.means = means
        assignements = km._step_E(points_data)
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
    
    Parameters
    ----------
    init : str
        The method with which the algorithm can be initialized.
        Must be in ['random','plus','AF_KMC','kmeans','GMM','VBGMM']
    
    n_components : int
        the number of clusters
        
    points_data : an array (n_points,dim)
    
    covariance_type : str
        Type of covariance : 'full' or 'spherical'
    
    Other Parameters
    ----------------
    points_test : array (n_points_bis,dim) | Optional
        Initializes using early stopping in order to avoid over fitting.
    
    Returns
    -------
    means : an array (n_components,dim)
        The initial means computed
    
    cov : an array (n_components,dim,dim)
        The initial covariances computed
        
    log_weights : an array (n_components,)
        The initial weights (log) computed
        
    """
    
    n_points,dim = points_data.shape
    
    log_weights = - np.log(n_components) * np.ones(n_components)
    
    # Warning : the algorithm is very sensitive to these first covariances given
    if covariance_type == "full":
        cov_init = np.cov(points_data.T)
        cov = np.tile(cov_init, (n_components,1,1))
    elif covariance_type == "spherical":
        cov_init = np.var(points_data, axis=0, ddof=1).mean()
        cov = cov_init * np.ones(n_components)
    
    if (init == "random"):
        means = initialization_random(n_components,points_data)
    elif(init == "plus"):
        means = initialization_plus_plus(n_components,points_data)
    elif(init == "AF_KMC"):
        means = initialization_AF_KMC(n_components,points_data)
    elif(init == "kmeans"):
        means,_ = initialization_k_means(n_components,points_data)
    elif(init == "GMM"):
        means,cov,log_weights,_ = initialization_GMM(n_components,points_data,points_test,covariance_type)
    elif(init == "VBGMM"):
        means,cov,log_weights,_ = initialization_VBGMM(n_components,points_data,points_test,covariance_type)
    
    return means,cov,log_weights