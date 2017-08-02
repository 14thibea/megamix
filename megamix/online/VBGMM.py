# -*- coding: utf-8 -*-
#
#Created on Fri Apr 14 13:37:08 2017
#
#author: Elina Thibeau-Sutre
#

from .base import BaseMixture
from .base import _log_normal_matrix
from .kmeans import dist_matrix
from megamix.batch.initializations import initialization_plus_plus

import numpy as np
from scipy.misc import logsumexp
import scipy

class VariationalGaussianMixture(BaseMixture):
    """
    Variational Bayesian Estimation of a Gaussian Mixture
    
    This class allows to infer an approximate posterior distribution over the
    parameters of a Gaussian mixture distribution.
    
    The weights distribution is a Dirichlet distribution with parameter alpha
    (see Bishop's book p474-486)
    
    Parameters
    ----------
    
    n_components : int, defaults to 1.
        Number of clusters used.
    
    init : str, defaults to 'kmeans'.
        Method used in order to perform the initialization,
        must be in ['random', 'plus', 'AF_KMC', 'kmeans', 'GMM'].  

    reg_covar : float, defaults to 1e-6
        In order to avoid null covariances this float is added to the diagonal
        of covariance matrices.                
    
    type_init : str, defaults to 'resp'.        
        The algorithm is initialized using this data (responsibilities if 'resp'
        or means, covariances and weights if 'mcw').

    Other parameters
    ----------------                
    
    alpha_0 : float, Optional | defaults to None.
        The prior parameter on the weight distribution (Dirichlet).
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
        
        If None, the value is set to the covariance of points_data
        
    Attributes
    ----------
    
    name : str
        The name of the method : 'VBGMM'
    
    alpha : array of floats (n_components,)
        Contains the parameters governing the weight distribution (Dirichlet)
    
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
    'Pattern Recognition and Machine Learning', Bishop
 
    """

    def __init__(self, n_components=1,alpha_0=None,beta_0=None,
                 nu_0=None,means_prior=None,cov_wishart_prior=None,
                 reg_covar=1e-6,kappa=1.0,n_jobs=1,window=1):
        
        super(VariationalGaussianMixture, self).__init__()

        self.name = 'VBGMM'
        self.n_components = n_components
        self.covariance_type = "full"
        self.reg_covar = reg_covar
        self.init = 'usual'
        
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.nu_0 = nu_0
        self._means_prior = means_prior
        self._inv_prec_prior = cov_wishart_prior
        self.kappa = kappa
        self.window = window
        self.n_jobs = n_jobs
        
        self._is_initialized = False
        self.iter = 0
        self.convergence_criterion_data = []
        self.convergence_criterion_test = []
        
        self._check_common_parameters()
        

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
#            diff = point.reshape(dim) - X[i]
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
        
        self.log_weights = logsumexp(log_resp,axis=0) + np.log(n_points)
        
    def _initialize(self,points):
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
        
        
        self._check_prior_parameters(points)
        
        
        if not self.init == 'read_and_init':
            # Parameters
            self.means = initialization_plus_plus(self.n_components,points)
            self._initialize_cov(points)
            
        # Computation of self.cov_chol
        self.cov_chol = np.empty(self.cov.shape)
        for i in range(self.n_components):
            self.cov_chol[i],inf = scipy.linalg.lapack.dpotrf(self.cov[i],lower=True)
                
        if not self.init == 'read_and_init':        
            self._initialize_weights(points)
            
            self.iter = n_points + 1
        
        
        # Hyperparameters
        weights = np.exp(self.log_weights)
        self.alpha = self.alpha_0 + weights
        self.beta = self.beta_0 + weights
        self.nu = self.nu_0 + weights

        # Sufficient statistics
        self.N = weights/n_points
        self.X = self.means * self.N[:,np.newaxis]
        self.S = self.cov * self.N[:,np.newaxis,np.newaxis]
        
        # Matrix W
        self._inv_prec = self.cov * self.nu[:,np.newaxis,np.newaxis]
        
        self.iter = n_points + 1
        self._is_initialized = True
        
        return self.n_components
        
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
          
        log_gaussian = _log_normal_matrix(points,self.means,self.cov_chol,self.covariance_type,self.n_jobs)
        digamma_sum = np.sum(scipy.special.psi(.5 * (self.nu - np.arange(0, dim)[:,np.newaxis])),0)
        log_lambda = digamma_sum + dim * np.log(2) + dim/self.beta
        
        log_prob = self.log_weights + log_gaussian + 0.5 * (log_lambda - dim * np.log(self.nu))
        
        log_prob_norm = logsumexp(log_prob, axis=1)
        log_resp = log_prob - log_prob_norm[:,np.newaxis]
                    
        return log_prob_norm,log_resp
    
    def _estimate_wishart_full(self,N,X,S):
        """
        This method computes the new value of _inv_prec with given parameteres
        (in the case of full covariances)
        
        Parameters
        ----------
        N : an array (n_components,)
            the empirical weights
        X_barre: an array (n_components,dim)
            the empirical means 
        S: an array (n_components,dim,dim)
            the empirical covariances
            
        """
        for i in range(self.n_components):
            diff = X[i] - self._means_prior
            product = self.beta_0 * N[i]/self.beta[i] * np.outer(diff,diff)
            self._inv_prec[i] = self._inv_prec_prior + N[i] * S[i] + product
            
    def _estimate_wishart_spherical(self,N,X,S):
        """
        This method computes the new value of _inv_prec with given parameteres
        (in the case of spherical covariances)
        
        Parameters
        ----------
        N : an array (n_components,)
            the empirical weights
        X_barre: an array (n_components,dim)
            the empirical means 
        S: an array (n_components,dim,dim)
            the empirical covariances
            
        """
        for i in range(self.n_components):
            diff = X[i] - self._means_prior
            product = self.beta_0 * N[i] / self.beta[i] * np.mean(np.square(diff), 1)
            self._inv_prec[i] = self._inv_prec_prior + N[i] * S[i] + product
        # To test
                
    def _step_M(self,log_resp):
        """
        In this step the algorithm updates the values of the parameters (means, covariances,
        alpha, beta, nu).
        
        Parameters
        ----------
        points : an array (n_points,dim)
        
        log_resp: an array (n_points,n_components)
            an array containing the logarithm of the responsibilities.
            
        """
        
        _,dim = self.X.shape
        
        weights = self.N * self.iter
        
        #Parameters update
        self.alpha = self.alpha_0 + weights
        self.beta = self.beta_0 + weights
        self.nu = self.nu_0 + weights
        
        # Weights update
        self.log_weights = scipy.special.psi(self.alpha) - scipy.special.psi(np.sum(self.alpha))
        
        # Means update
        means = self.X / self.N[:,np.newaxis]
        self.means = (self.beta_0 * self._means_prior + weights[:,np.newaxis] * means) / self.beta[:, np.newaxis]
        
        # Covariance update
        cov = self.S / self.N[:,np.newaxis,np.newaxis]
        if self.covariance_type=="full":
            self._estimate_wishart_full(weights,means,cov)
            self.cov = self._inv_prec / self.nu[:,np.newaxis,np.newaxis]
            for i in range(self.n_components):
                self.cov_chol[i],err = scipy.linalg.lapack.dpotrf(self.cov[i],lower=True)
            if err:
                raise ValueError('Error while computing the cholesky factorization')
            
        elif self.covariance_type=="spherical":
            self._estimate_wishart_spherical(weights,means,cov)
            self.cov = self._inv_prec / self.nu
            self.cov_chol = np.sqrt(self.cov)
            
    def _sufficient_statistics(self,points,log_resp):
        
        n_points,dim = points.shape
        resp = np.exp(log_resp)
                 
        # New sufficient statistics
        N = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        N /= n_points
        
        X = np.dot(resp.T,points)
        X /= n_points
        
        S = np.zeros((self.n_components,dim,dim))
        for i in range(self.n_components):
#            diff = point.reshape(dim) - X[i]
            diff = points - self.means[i]
            diff_weighted = diff * resp[:,i:i+1]
            S[i] = np.dot(diff_weighted.T,diff)
            S[i].flat[::dim+1] += self.reg_covar
        S /= n_points
        
        # Sufficient statistics update
        gamma = 1/((self.iter + n_points//2)**self.kappa)
        
        self.N = (1-gamma)*self.N + gamma*N
        self.X = (1-gamma)*self.X + gamma*X
        self.S = (1-gamma)*self.S + gamma*S
    
                
#    def _convergence_criterion_simplified(self,points,log_resp,log_prob_norm):
#        """
#        Compute the lower bound of the likelihood using the simplified Bishop's
#        book formula. Can only be used with data which fits the model.
#        
#        
#        Parameters
#        ----------
#        points : an array (n_points,dim)
#        
#        log_resp: an array (n_points,n_components)
#            an array containing the logarithm of the responsibilities.
#            
#        log_prob_norm : an array (n_points,)
#            logarithm of the probability of each sample in points
#        
#        Returns
#        -------
#        result : float
#            the lower bound of the likelihood
#            
#        """
#        
#        resp = np.exp(log_resp)
#        n_points,dim = points.shape
#        
#        prec = np.linalg.inv(self._inv_prec)
#        prec_prior = np.linalg.inv(self._inv_prec_prior)
#        
#        lower_bound = np.zeros(self.n_components)
#        
#        for i in range(self.n_components):
#            
#            lower_bound[i] = _log_B(prec_prior,self.nu_0) - _log_B(prec[i],self.nu[i])
#            
#            resp_i = resp[:,i:i+1]
#            log_resp_i = log_resp[:,i:i+1]
#            
#            lower_bound[i] -= np.sum(resp_i*log_resp_i)
#            lower_bound[i] += dim*0.5*(np.log(self.beta_0) - np.log(self.beta[i]))
#        
#        result = np.sum(lower_bound)
#        result += _log_C(self.alpha_0 * np.ones(self.n_components)) - _log_C(self.alpha)
#        result -= n_points * dim * 0.5 * np.log(2*np.pi)
#        
#        return result
#        
#    def _convergence_criterion(self,points,log_resp,log_prob_norm):
#        """
#        Compute the lower bound of the likelihood using the Bishop's book formula.
#        The formula cannot be simplified (as it is done in scikit-learn) as we also
#        use it to calculate the lower bound of test points, in this case no
#        simplification can be done.
#          
#        
#        Parameters
#        ----------
#        points : an array (n_points,dim)
#        
#        log_resp: an array (n_points,n_components)
#            an array containing the logarithm of the responsibilities.
#            
#        log_prob_norm : an array (n_points,)
#            logarithm of the probability of each sample in points
#        
#        Returns
#        -------
#        result : float
#            the lower bound of the likelihood
#            
#        """
#        
#        resp = np.exp(log_resp)
#        n_points,dim = points.shape
#        
#        # Convenient statistics
#        N = np.exp(logsumexp(log_resp,axis=0)) + 10*np.finfo(resp.dtype).eps    #Array (n_components,)
#        X_barre = np.tile(1/N, (dim,1)).T * np.dot(resp.T,points)               #Array (n_components,dim)
#        S = _full_covariance_matrices(points,X_barre,N,resp,self.reg_covar,self.n_jobs)
#        
#        prec = np.linalg.inv(self._inv_prec)
#        prec_prior = np.linalg.inv(self._inv_prec_prior)
#        
#        lower_bound = np.zeros(self.n_components)
#        
#        for i in range(self.n_components):
#        
#            digamma_sum = np.sum(scipy.special.psi(.5 * (self.nu[i] - np.arange(0, dim)[:,np.newaxis])),0)
#            log_det_prec_i = digamma_sum + dim * np.log(2) - self._log_det_inv_prec[i] #/!\ Inverse
#            
#            #First line
#            lower_bound[i] = log_det_prec_i - dim/self.beta[i] - self.nu[i]*np.trace(np.dot(S[i],prec[i]))
#            diff = X_barre[i] - self.means[i]
#            lower_bound[i] += -self.nu[i]*np.dot(diff,np.dot(prec[i],diff.T))
#            lower_bound[i] *= 0.5 * N[i]
#            
#            #Second line
#            lower_bound[i] += (self.alpha_0 - self.alpha[i]) * self.log_weights[i]
#            lower_bound[i] += _log_B(prec_prior,self.nu_0) - _log_B(prec[i],self.nu[i])
#            
#            resp_i = resp[:,i:i+1]
#            log_resp_i = log_resp[:,i:i+1]
#            
#            lower_bound[i] += np.sum(resp_i) * self.log_weights[i] - np.sum(resp_i*log_resp_i)
#            lower_bound[i] += 0.5 * (self.nu_0 - self.nu[i]) * log_det_prec_i
#            lower_bound[i] += dim*0.5*(np.log(self.beta_0) - np.log(self.beta[i]))
#            lower_bound[i] += dim*0.5*(1 - self.beta_0/self.beta[i] + self.nu[i])
#            
#            #Third line without the last term which is not summed
#            diff = self.means[i] - self._means_prior
#            lower_bound[i] += -0.5*self.beta_0*self.nu[i]*np.dot(diff,np.dot(prec[i],diff.T))
#            lower_bound[i] += -0.5*self.nu[i]*np.trace(np.dot(self._inv_prec_prior,prec[i]))
#                
#        result = np.sum(lower_bound)
#        result += _log_C(self.alpha_0 * np.ones(self.n_components))- _log_C(self.alpha)
#        result -= n_points * dim * 0.5 * np.log(2*np.pi)
#        
#        return result
    
    
#    def _get_parameters(self):
#        return (self.log_weights, self.means, self.cov,
#                self.alpha, self.beta, self.nu)
#    
#
#    def _set_parameters(self, params,verbose=True):
#        (self.log_weights, self.means, self.cov,
#        self.alpha, self.beta, self.nu )= params
#         
#        # Matrix W
#        self._inv_prec = self.cov * self.nu[:,np.newaxis,np.newaxis]
#        self._log_det_inv_prec = np.log(np.linalg.det(self._inv_prec))
#        if self.n_components != len(self.means) and verbose:
#            print('The number of components changed')
#        self.n_components = len(self.means)
        
            
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
#        idx_existing = np.where(exist==1)
#        
#        log_weights = self.log_weights[idx_existing]
#        means = self.means[idx_existing]
#        cov = self.cov[idx_existing]
#        alpha = self.alpha[idx_existing]
#        beta = self.beta[idx_existing]
#        nu = self.nu[idx_existing]
#                
#        params = (log_weights, means, cov,
#                  alpha, beta, nu)
#        
#        return params
