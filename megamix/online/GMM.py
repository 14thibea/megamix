#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#Created on Mon Apr 10 11:34:50 2017
#
#author: Elina Thibeau-Sutre
#

from .base import BaseMixture
from .base import _log_normal_matrix
from megamix.batch.initializations import initialization_plus_plus
from .kmeans import dist_matrix

import numpy as np
from scipy.misc import logsumexp
import scipy
from choldate import cholupdate


class GaussianMixture(BaseMixture):
    """
    Gaussian Mixture Model
    
    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.
    
    Parameters
    ----------
    
    n_components : int, defaults to 1.
        Number of clusters used.
    
    init : str, defaults to 'kmeans'.
        Method used in order to perform the initialization,
        must be in ['random', 'plus', 'AF_KMC', 'kmeans'].  

    reg_covar : float, defaults to 1e-6
        In order to avoid null covariances this float is added to the diagonal
        of covariance matrices.                
    
    type_init : str, defaults to 'resp'.        
        The algorithm is initialized using this data (responsibilities if 'resp'
        or means, covariances and weights if 'mcw').

    Attributes
    ----------
    
    name : str
        The name of the method : 'GMM'
    
    cov : array of floats (n_components,dim,dim)
        Contains the computed covariance matrices of the mixture.
    
    means : array of floats (n_components,dim)
        Contains the computed means of the mixture.
    
    log_weights : array of floats (n_components,)
        Contains the logarithm of weights of each cluster.
    
    iter : int
        The number of iterations computed with the method fit()
    
    convergence_criterion_data : array of floats (iter,)
        Stores the value of the convergence criterion computed with data
        on which the model is fitted.
    
    convergence_criterion_test : array of floats (iter,) | if _early_stopping only
        Stores the value of the convergence criterion computed with test data
        if it exists.
    
    _is_initialized : bool
        Ensures that the method _initialize() has been used before using other
        methods such as score() or predict_log_assignements().
    
    Raises
    ------
    ValueError : if the parameters are inconsistent, for example if the cluster number is negative, init_type is not in ['resp','mcw']...
    
    References
    ----------
    'Pattern Recognition and Machine Learning', Bishop
 
    """

    def __init__(self, n_components=1,covariance_type="full",
                 kappa=1.0,reg_covar=1e-6,n_jobs=1,window=1,
                 update=False):
        
        super(GaussianMixture, self).__init__()

        self.name = 'GMM'
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.n_jobs = n_jobs
        self.kappa = kappa
        self.window = window
        self.update = update
        self.init = 'usual'
        
        self._is_initialized = False
        self.iter = 0
        self.convergence_criterion_data = []
        self.convergence_criterion_test = []
        
        self._check_common_parameters()
        self._check_parameters()

    def _check_parameters(self):
            
        if self.covariance_type not in ['full','spherical']:
            raise ValueError("Invalid value for 'init': %s "
                             "'covariance_type' should be in "
                             "['full', 'spherical']"
                             % self.covariance_type)
            
    def _initialize_cov(self,points):
        
        n_points,dim = points.shape
        assignements = np.zeros((n_points,self.n_components))
        
        M = dist_matrix(points,self.means)
        for i in range(n_points):
            index_min = np.argmin(M[i]) #the cluster number of the ith point is index_min
            if (isinstance(index_min,np.int64)):
                assignements[i][index_min] = 1
            else: #Happens when two points are equally distant from a cluster mean
                assignements[i][index_min[0]] = 1
        
        S = np.zeros((self.n_components,dim,dim))
        for i in range(self.n_components):
            diff = points - self.means[i]
            diff_weighted = diff * assignements[:,i:i+1]
            S[i] = np.dot(diff_weighted.T,diff)
            S[i].flat[::dim+1] += self.reg_covar
        S /= n_points
        
        self.cov = S * self.n_components
        
        
    def _initialize_weights(self,points):
        n_points,_ = points.shape
        
        log_prob = _log_normal_matrix(points,self.means,self.cov_chol,
                                      self.covariance_type,self.n_jobs)
        log_prob_norm = logsumexp(log_prob, axis=1)
        log_resp = log_prob - log_prob_norm[:,np.newaxis]
        
        self.log_weights = logsumexp(log_resp,axis=0) - np.log(n_points)
        
        
    def initialize(self,points):
        """
        This method initializes the Gaussian Mixture by setting the values of
        the means, covariances and weights.
        
        Parameters
        ----------
        points_data : an array (n_points,dim)
            Data on which the model is fitted.
        points_test: an array (n_points,dim) | Optional
            Data used to do early stopping (avoid overfitting)
            
        """
        
        n_points,dim = points.shape
        
        if self.init == 'usual':
            self.means = initialization_plus_plus(self.n_components,points)
            self.iter = n_points + 1
        if self.init in ['usual','read_kmeans']:
            self._initialize_cov(points)
            
        # Computation of self.cov_chol
        self.cov_chol = np.empty(self.cov.shape)
        for i in range(self.n_components):
            self.cov_chol[i],inf = scipy.linalg.lapack.dpotrf(self.cov[i],lower=True)
                
        if self.init in ['usual','read_kmeans']:        
            self._initialize_weights(points)

        weights = np.exp(self.log_weights)
        self.N = weights
        self.X = self.means * self.N[:,np.newaxis]
        self.S = self.cov * self.N[:,np.newaxis,np.newaxis]
        
        # Computation of S_chol if update=True
        if self.update:
            if self.covariance_type == 'full':
                self.S_chol = np.empty(self.S.shape)
                for i in range(self.n_components):
                    self.S_chol[i],inf = scipy.linalg.lapack.dpotrf(self.S[i],lower=True)
            elif self.covariance_type == 'spherical':
                self.S_chol = np.sqrt(self.S)
        
        self._is_initialized = True
        
        
    def _step_E(self, points):
        """
        In this step the algorithm evaluates the responsibilities of each points in each cluster
        
        Parameters
        ----------
        points : an array (n_points,dim)
        
        Returns
        -------
        log_resp: an array (n_points,n_components)
            an array containing the logarithm of the responsibilities.
        log_prob_norm : an array (n_points,)
            logarithm of the probability of each sample in points
            
        """
        log_normal_matrix = _log_normal_matrix(points,self.means,self.cov_chol,
                                               self.covariance_type,self.n_jobs)
        log_product = log_normal_matrix + self.log_weights
        log_prob_norm = logsumexp(log_product,axis=1)

        log_resp = log_product - log_prob_norm[:,np.newaxis]
        
        return log_prob_norm,log_resp
      
    def _step_M(self):
        """
        In this step the algorithm updates the values of the parameters (means, covariances,
        alpha, beta, nu).
        
        Parameters
        ----------
        points : an array (n_points,dim)
        
        log_resp: an array (n_points,n_components)
            an array containing the logarithm of the responsibilities.
            
        """        
        self.log_weights = np.log(self.N)
        self.means = self.X / self.N[:,np.newaxis]
        self.cov = self.S / self.N[:,np.newaxis,np.newaxis]
        if self.update:
            self.cov_chol = self.S_chol/np.sqrt(self.N)[:,np.newaxis,np.newaxis]
        else:
            for i in range(self.n_components):
                self.cov_chol[i],inf = scipy.linalg.lapack.dpotrf(self.cov[i],lower=True)
        
        
    def _sufficient_statistics(self,points,log_resp):
        
        n_points,dim = points.shape
        resp = np.exp(log_resp)
        
        gamma = 1/(((self.iter + n_points)//2)**self.kappa)
        
        # New sufficient statistics
        N = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        N /= n_points
        
        X = np.dot(resp.T,points)
        X /= n_points
        
        S = np.empty((self.n_components,dim,dim))
        for i in range(self.n_components):
            diff = points - self.means[i]
            diff_weighted = diff * np.sqrt(resp[:,i:i+1])
            S[i] = np.dot(diff_weighted.T,diff_weighted)

            if self.update:
                # diff_weighted is recquired in order to update cov_chol, so we begin
                # its update here
                u = np.sqrt(gamma/((1-gamma)*n_points)) * diff_weighted
                for j in range(n_points):
                    cholupdate(self.S_chol[i].T,u[j])
                
#                u = np.sqrt(gamma/((1-gamma)*self.N[i]))*diff_weighted
#                for j in range(n_points):
#                    cholupdate(self.cov_chol[i].T,u[j])
            
        S /= n_points
        if self.update:
            self.S_chol *= np.sqrt((1-gamma))
        
        # Sufficient statistics update
        self.N = (1-gamma)*self.N + gamma*N
        self.X = (1-gamma)*self.X + gamma*X
        self.S = (1-gamma)*self.S + gamma*S
    
    
    def _convergence_criterion(self,points,_,log_prob_norm):
        """
        Compute the log likelihood.
        
        
        Parameters
        ----------
        points : an array (n_points,dim)
            
        log_prob_norm : an array (n_points,)
            logarithm of the probability of each sample in points
        
        Returns
        -------
        result : float
            the log likelihood
            
        """
        return np.sum(log_prob_norm)

    
    def _get_parameters(self):
        return (self.N, self.X, self.S)
    

    def _set_parameters(self, params,verbose=True):
        self.N, self.X, self.S = params
        
        real_components = len(self.X)
        if self.n_components != real_components and verbose:
            print('The number of components changed')
        self.n_components = real_components
        
            
#    def _limiting_model(self,points):
#        
#        n_points,dim = points.shape
#        log_resp = self.predict_log_resp(points)
#        _,n_components = log_resp.shape
#    
#        exist = np.zeros(n_components)
#        
#        for i in range(n_points):
#            for j in range(n_components):
#                if np.argmax(log_resp[i])==j:
#                    exist[j] = 1
#        
#
#        idx_existing = np.where(exist==1)
#        
#        log_weights = self.log_weights[idx_existing]
#        means = self.means[idx_existing]
#        cov = self.cov[idx_existing]
#                
#        params = (log_weights, means, cov)
#        
#        return params