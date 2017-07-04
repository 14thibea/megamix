# -*- coding: utf-8 -*-
#
#Created on Fri Apr 14 15:21:17 2017
#
#author: Elina Thibeau-Sutre
#

from .initializations import initialize_log_assignements,initialize_mcw
from .base import _full_covariance_matrices
from .base import _log_normal_matrix
from .base import BaseMixture
from .base import _log_B

import numpy as np
from scipy.special import psi,betaln
from scipy.misc import logsumexp

class XPVariationalGaussianMixture(BaseMixture):

    """
    Variational Bayesian Estimation of a Gaussian Mixture with Dirichlet Process
    
    This class allows to infer an approximate posterior distribution over the
    parameters of a Gaussian mixture distribution.
    
    The weights distribution follows a Dirichlet Process with attribute alpha.
    
    Parameters
    ----------
    
    n_components : int, defaults to 1.
        Number of clusters used.
    
    init : str, defaults to 'kmeans'.
        Method used in order to perform the initialization,
        must be in ['random', 'plus', 'AF_KMC', 'kmeans', 'GMM', 'VBGMM'].

    reg_covar : float, defaults to 1e-6
        In order to avoid null covariances this float is added to the diagonal
        of covariance matrices.
    
    type_init : str, defaults to 'resp'.        
        The algorithm is initialized using this data (responsibilities if 'resp'
        or means, covariances and weights if 'mcw').

    Other parameters
    ----------------                
    
    alpha_0 : float, Optional | defaults to None.
        The prior parameter on the weight distribution (Beta).
        A high value of alpha_0 will lead to equal weights, while a low value
        will allow some clusters to shrink and disappear. Must be greater than 0.
    
        If None, the value is set to 1/n_components                         
    
    beta_0 : float, Optional | defaults to None.
        The precision prior on the mean distribution (Gaussian).
        Must be greater than 0.
    
        If None, the value is set to 1.0                         
    
    nu_0 : float, Optional | defaults to None.
        The prior of the number of degrees of freedom on the covariance
        distributions (Wishart). Must be greater or equal to dim.
    
        If None, the value is set to dim
        
    means_prior : array (dim,), Optional | defaults to None
        The prior value to compute the value of the means.
        
        If None, the value is set to the mean of points_data
        
    cov_wishart_prior : type depends on covariance_type, Optional | defaults to None
        If covariance_type is 'full' type must be array (dim,dim)
        If covariance_type is 'spherical' type must be float
        The prior value to compute the value of the precisions.
        
    pypcoeff : float | defaults to 0
        If 0 the weights are generated according to a Dirichlet Process
        If >0 and <=1 the weights are generated according to a Pitman-Yor
        Process.

    Attributes
    ----------
    
    name : str
        The name of the method : 'VBGMM'
    
    alpha : array of floats (n_components,2)
        Contains the parameters of the weight distribution (Beta)
    
    beta : array of floats (n_components,)
        Contains coefficients which are multipied with the precision matrices
        to form the precision matrix on the Gaussian distribution of the means.    
    
    nu : array of floats (n_components,)
        Contains the number of degrees of freedom on the distribution of
        covariance matrices.
    
    _inv_prec : array of floats (n_components,dim,dim)
        Contains the equivalent of the matrix W described in Bishop's book. It
        is proportional to cov.
    
    _log_det_inv_prec : array of floats (n_components,)
        Contains the logarithm of the determinant of W matrices.
    
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
    'Variational Inference for Dirichlet Process Mixtures', D. Blei and M. Jordan
 
    """
    
    def __init__(self, n_components=1,init="kmeans",alpha_0=None,beta_0=None,
                 nu_0=None,means_prior=None,cov_wishart_prior=None,
                 reg_covar=1e-6,type_init='resp',n_jobs=1,pypcoeff=0):
        
        super(XPVariationalGaussianMixture, self).__init__()
        
        self.name = 'DPGMM'
        self.n_components = n_components
        self.covariance_type = "full"
        self.init = init
        self.type_init = type_init
        self.reg_covar = reg_covar
        
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.nu_0 = nu_0
        self.pypcoeff = pypcoeff
        self._means_prior = means_prior
        self._inv_prec_prior = cov_wishart_prior
        self.n_jobs = n_jobs
        
        self._is_initialized = False
        self.iter = 0
        self.convergence_criterion_data = []
        self.convergence_criterion_test = []
        
        self._check_common_parameters()
        self._check_parameters()

    def _check_parameters(self):
        """
        Check the value of the init parameter
        
        """
        
        if self.init not in ['random', 'plus', 'kmeans', 'AF_KMC', 'GMM', 'VBGMM']:
            raise ValueError("Invalid value for 'init': %s "
                             "'init' should be in "
                             "['random','plus','kmeans','AF_KMC','GMM','VBGMM']"
                             % self.init)
            
        if self.pypcoeff < 0 or self.pypcoeff > 1:
            raise ValueError("Invalid value for 'pypcoeff': %s "
                             "'pypcoeff' should be between 0 and 1"
                             % self.init)
          
    def _initialize(self,points_data,points_test=None):
        """
        This method initializes the Variational Gaussian Mixture by setting the values
        of the means, the covariances and other specific parameters (alpha, beta, nu)
        
        Parameters
        ----------
        points_data : an array (n_points,dim)
            Data on which the model is fitted.
        points_test: an array (n_points,dim) | Optional
            Data used to do early stopping (avoid overfitting)
            
        """
        
        n_points,dim = points_data.shape

        self._check_prior_parameters(points_data)
		
        if self.type_init == 'resp':
            log_assignements = initialize_log_assignements(self.init,self.n_components,points_data,
                                                           points_test)
            self._inv_prec = np.empty((self.n_components,dim,dim))
            self._log_det_inv_prec = np.empty(self.n_components)
            self.cov = np.empty((self.n_components,dim,dim))
            self.alpha = np.empty((self.n_components,2))
            self.log_weights = np.empty(self.n_components)
            self._step_M(points_data,log_assignements)
        
        elif self.type_init == 'mcw':
            #Means, covariances and weights
            means,cov,log_weights = initialize_mcw(self.init,self.n_components,points_data,
                                                   points_test)
            self.cov = cov
            self.means = means
            self.log_weights = log_weights
            
            # Hyper parameters
            N = np.exp(log_weights) * n_points
            self.alpha = np.asarray([1 + N,
                          self.alpha_0 + np.hstack((np.cumsum(N[::-1])[-2::-1], 0))]).T
            self.alpha += np.asarray([-self.pypcoeff * np.ones(self.n_components),
                                      self.pypcoeff * np.arange(self.n_components)]).T
            self.beta = self.beta_0 + N
            self.nu = self.nu_0 + N
            
            # Matrix W
            self._inv_prec = cov * self.nu[:,np.newaxis,np.newaxis]
            self._log_det_inv_prec = np.log(np.linalg.det(self._inv_prec))
            
        elif self.type_init == 'user':
            
            # Hyper parameters
            N = np.exp(self.log_weights) * n_points
            self.alpha = np.asarray([1 + N,
                          self.alpha_0 + np.hstack((np.cumsum(N[::-1])[-2::-1], 0)) ]).T
            self.alpha += np.asarray([-self.pypcoeff * np.ones(self.n_components),
                                      self.pypcoeff * np.arange(self.n_components)]).T
            self.beta = self.beta_0 + N
            self.nu = self.nu_0 + N
            
            # Matrix W
            self._inv_prec = self.cov * self.nu[:,np.newaxis,np.newaxis]
            self._log_det_inv_prec = np.log(np.linalg.det(self._inv_prec))
            
            
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
        
        n_points,dim = points.shape
        
        log_gaussian = _log_normal_matrix(points,self.means,self.cov,'full',self.n_jobs)
        digamma_sum = np.sum(psi(.5 * (self.nu - np.arange(0, dim)[:,np.newaxis])),0)
        log_lambda = digamma_sum + dim * np.log(2) + dim/self.beta
        
        log_prob = self.log_weights + log_gaussian + 0.5 * (log_lambda - dim * np.log(self.nu))
        
        log_prob_norm = logsumexp(log_prob, axis=1)
        log_resp = log_prob - log_prob_norm[:,np.newaxis]
                    
        return log_prob_norm,log_resp
    
    def _step_M(self,points,log_resp):
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
        
        resp = np.exp(log_resp)
        
        # Convenient statistics
        N = np.sum(resp,axis=0) + 10*np.finfo(resp.dtype).eps            #Array (n_components,)
        X_barre = 1/N[:,np.newaxis] * np.dot(resp.T,points)              #Array (n_components,dim)
        S = _full_covariance_matrices(points,X_barre,N,resp,self.reg_covar,self.n_jobs)
        
        #Parameters update
        self.alpha = np.asarray([1.0 + N,
                                  self.alpha_0 + np.hstack((np.cumsum(N[::-1])[-2::-1], 0))]).T
        self.alpha += np.asarray([-self.pypcoeff * np.ones(self.n_components),
                                      self.pypcoeff * np.arange(self.n_components)]).T
        self.beta = self.beta_0 + N
        self.nu = self.nu_0 + N
        
        # Weights update
        for i in range(self.n_components):
            if i==0:
                self.log_weights[i] = psi(self.alpha[i][0]) - psi(np.sum(self.alpha[i]))
            else:
                self.log_weights[i] = psi(self.alpha[i][0]) - psi(np.sum(self.alpha[i]))
                self.log_weights[i] += self.log_weights[i-1] + psi(self.alpha[i-1][1]) - psi(self.alpha[i-1][0])
        
        # Means update
        means = self.beta_0 * self._means_prior + N[:,np.newaxis] * X_barre
        self.means = means * np.reciprocal(self.beta)[:,np.newaxis]
        self.means_estimated = self.means
        
        # Covariance update
        for i in range(self.n_components):
            diff = X_barre[i] - self._means_prior
            product = self.beta_0 * N[i]/self.beta[i] * np.outer(diff,diff)
            self._inv_prec[i] = self._inv_prec_prior + N[i] * S[i] + product
            
            det_inv_prec = np.linalg.det(self._inv_prec[i])
            self._log_det_inv_prec[i] = np.log(det_inv_prec)
            self.cov[i] = self._inv_prec[i] / self.nu[i]
        
    def _convergence_criterion_simplified(self,points,log_resp,log_prob_norm):
        """
        Compute the lower bound of the likelihood using the simplified Blei and
        Jordan formula. Can only be used with data which fits the model.
        
        
        Parameters
        ----------
        points : an array (n_points,dim)
        
        log_resp: an array (n_points,n_components)
            an array containing the logarithm of the responsibilities.
            
        log_prob_norm : an array (n_points,)
            logarithm of the probability of each sample in points
        
        Returns
        -------
        result : float
            the lower bound of the likelihood
            
        """
        
        resp = np.exp(log_resp)
        n_points,dim = points.shape
        
        prec = np.linalg.inv(self._inv_prec)
        prec_prior = np.linalg.inv(self._inv_prec_prior)
        
        lower_bound = np.zeros(self.n_components)
        
        for i in range(self.n_components):
            
            lower_bound[i] = _log_B(prec_prior,self.nu_0) - _log_B(prec[i],self.nu[i])
            
            resp_i = resp[:,i:i+1]
            log_resp_i = log_resp[:,i:i+1]
            
            lower_bound[i] -= np.sum(resp_i*log_resp_i)
            lower_bound[i] += dim*0.5*(np.log(self.beta_0) - np.log(self.beta[i]))
        
        result = np.sum(lower_bound)
        result -= self.n_components * betaln(1,self.alpha_0)
        result += np.sum(betaln(self.alpha.T[0],self.alpha.T[1]))
        result -= n_points * dim * 0.5 * np.log(2*np.pi)
        
        return result
    
    
    def _convergence_criterion(self,points,log_resp,log_prob_norm):
        """
        Compute the lower bound of the likelihood using the Blei and Jordan formula.
        The formula cannot be simplified (as it is done in scikit-learn) as we also
        use it to calculate the lower bound of test points, in this case no
        simplification can be done.
          
        
        Parameters
        ----------
        points : an array (n_points,dim)
        
        log_resp: an array (n_points,n_components)
            an array containing the logarithm of the responsibilities.
            
        log_prob_norm : an array (n_points,)
            logarithm of the probability of each sample in points
        
        Returns
        -------
        result : float
            the lower bound of the likelihood
            
        """
        
        resp = np.exp(log_resp)
        n_points,dim = points.shape
        
        # Convenient statistics
        N = np.sum(resp,axis=0) + 10*np.finfo(resp.dtype).eps               #Array (n_components,)
        X_barre = 1/N[:,np.newaxis] * np.dot(resp.T,points)                 #Array (n_components,dim)
        S = _full_covariance_matrices(points,X_barre,N,resp,self.reg_covar,self.n_jobs)
             
        prec = np.linalg.inv(self._inv_prec)
        prec_prior = np.linalg.inv(self._inv_prec_prior)
        
        lower_bound = np.zeros(self.n_components)
        
        for i in range(self.n_components):
            
            digamma_sum = np.sum(psi(.5 * (self.nu[i] - np.arange(0, dim)[:,np.newaxis])),0)
            log_det_prec_i = digamma_sum + dim * np.log(2) - self._log_det_inv_prec[i] #/!\ Inverse
            
            #First line
            lower_bound[i] = log_det_prec_i - dim/self.beta[i] - self.nu[i]*np.trace(np.dot(S[i],prec[i]))
            diff = X_barre[i] - self.means[i]
            lower_bound[i] += -self.nu[i]*np.dot(diff,np.dot(prec[i],diff.T))
            lower_bound[i] *= 0.5 * N[i]
            
            #Second line
            lower_bound[i] += _log_B(prec_prior,self.nu_0) - _log_B(prec[i],self.nu[i])
            
            resp_i = resp[:,i:i+1]
            log_resp_i = log_resp[:,i:i+1]
            
            lower_bound[i] -= np.sum(resp_i*log_resp_i)
            lower_bound[i] += 0.5 * (self.nu_0 - self.nu[i]) * log_det_prec_i
            lower_bound[i] += dim*0.5*(np.log(self.beta_0) - np.log(self.beta[i]))
            lower_bound[i] += dim*0.5*(1 - self.beta_0/self.beta[i] + self.nu[i])
            
            #Third line without the last term which is not summed
            diff = self.means[i] - self._means_prior
            lower_bound[i] += -0.5*self.beta_0*self.nu[i]*np.dot(diff,np.dot(prec[i],diff.T))
            lower_bound[i] += -0.5*self.nu[i]*np.trace(np.dot(self._inv_prec_prior,prec[i]))
            
            #Terms with alpha
            lower_bound[i] += (N[i] + 1.0 - self.alpha[i,0]) * (psi(self.alpha[i,0]) - psi(np.sum(self.alpha[i])))
            lower_bound[i] += (np.sum(N[i+1::]) + self.alpha_0 - self.alpha[i,1]) * (psi(self.alpha[i,1]) - psi(np.sum(self.alpha[i])))
        
        result = np.sum(lower_bound)
        result -= self.n_components * betaln(1,self.alpha_0)
        result += np.sum(betaln(self.alpha.T[0],self.alpha.T[1]))
        result -= n_points * dim * 0.5 * np.log(2*np.pi)
        
        return result

    
    def _get_parameters(self):
        return (self.log_weights, self.means, self.cov,
                self.alpha, self.beta, self.nu)
    

    def _set_parameters(self, params,verbose=True):
        (self.log_weights, self.means, self.cov,
        self.alpha, self.beta, self.nu )= params
         
        # Matrix W
        self._inv_prec = self.cov * self.nu[:,np.newaxis,np.newaxis]
        self._log_det_inv_prec = np.log(np.linalg.det(self._inv_prec))
        if self.n_components != len(self.means) and verbose:
            print('The number of components changed')
        self.n_components = len(self.means)
        
    def _limiting_model(self,points):
        
        n_points,dim = points.shape
        log_resp = self.predict_log_resp(points)
        _,n_components = log_resp.shape
    
        exist = np.zeros(n_components)
        
        for i in range(n_points):
            for j in range(n_components):
                if np.argmax(log_resp[i])==j:
                    exist[j] = 1
        
        idx_existing = np.where(exist==1)
        
        log_weights = self.log_weights[idx_existing]
        means = self.means[idx_existing]
        cov = self.cov[idx_existing]
        alpha = self.alpha[idx_existing]
        beta = self.beta[idx_existing]
        nu = self.nu[idx_existing]
        
        params = (log_weights, means, cov,
                  alpha, beta, nu)
        
        return params