# -*- coding: utf-8 -*-
#
#Created on Mon Apr 10 11:34:50 2017
#
#author: Elina Thibeau-Sutre
#

from .base import BaseMixture
from .base import _log_normal_matrix
from .base import _full_covariance_matrix
from .base import _spherical_covariance_matrix
from .initializations import initialize_log_assignements,initialize_mcw

import numpy as np
from scipy.misc import logsumexp

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
        must be in ['random','plus','AF_KMC','kmeans','GMM'].  

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

    def __init__(self, n_components=1,covariance_type="full",init="kmeans",
                 reg_covar=1e-6,type_init='resp'):
        
        super(GaussianMixture, self).__init__()

        self.name = 'GMM'
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.init = init
        self.type_init = type_init
        self.reg_covar = reg_covar
        
        self._is_initialized = False
        self.iter = 0
        self.convergence_criterion_data = []
        self.convergence_criterion_test = []
        
        self._check_common_parameters()
        self._check_parameters()

    def _check_parameters(self):
        
        if self.init not in ['random', 'plus', 'kmeans', 'AF_KMC']:
            raise ValueError("Invalid value for 'init': %s "
                             "'init' should be in "
                             "['random', 'plus', 'kmeans', 'AF_KMC']"
                             % self.init)
            
        if self.covariance_type not in ['full','spherical']:
            raise ValueError("Invalid value for 'init': %s "
                             "'covariance_type' should be in "
                             "['full', 'spherical']"
                             % self.covariance_type)
            
    def _initialize(self,points_data,points_test=None):
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
        
        if self.type_init=='resp':
            log_assignements = initialize_log_assignements(self.init,self.n_components,points_data,points_test,self.covariance_type)
            self._step_M(points_data,log_assignements)
        elif self.type_init=='mcw':
            means,cov,log_weights = initialize_mcw(self.init,self.n_components,points_data,points_test,self.covariance_type)
            self.means = means
            self.cov = cov
            self.log_weights = log_weights
            
        self._is_initialized = True
    
    def _step_E(self,points):
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
        log_normal_matrix = _log_normal_matrix(points,self.means,self.cov,self.covariance_type)
        log_product = log_normal_matrix + self.log_weights[:,np.newaxis].T
        log_prob_norm = logsumexp(log_product,axis=1)
        
        log_resp = log_product - log_prob_norm[:,np.newaxis]
        
        return log_prob_norm,log_resp
      
    def _step_M(self,points,log_assignements):
        """
        In this step the algorithm updates the values of the parameters (means, covariances,
        alpha, beta, nu).
        
        Parameters
        ----------
        points : an array (n_points,dim)
        
        log_resp: an array (n_points,n_components)
            an array containing the logarithm of the responsibilities.
            
        """
        n_points,dim = points.shape
        
        assignements = np.exp(log_assignements)
        
        #Phase 1:
        product = np.dot(assignements.T,points)
        weights = np.sum(assignements,axis=0) + 10 * np.finfo(assignements.dtype).eps
        
        self.means = product / weights[:,np.newaxis]
        
        #Phase 2:
        if self.covariance_type=="full":
            self.cov = _full_covariance_matrix(points,self.means,weights,log_assignements,self.reg_covar)
        elif self.covariance_type=="spherical":
            self.cov = _spherical_covariance_matrix(points,self.means,weights,assignements,self.reg_covar)
                        
        #Phase 3:
        self.log_weights = logsumexp(log_assignements, axis=0) - np.log(n_points)
        
    def _convergence_criterion_simplified(self,points,_,log_prob_norm):
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
        return (self.log_weights, self.means, self.cov)
    

    def _set_parameters(self, params):
        self.log_weights, self.means, self.cov = params