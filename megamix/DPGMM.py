# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:21:17 2017

:author: Elina Thibeau-Sutre
"""

import utils
import initializations as initial
from base import _log_normal_matrix
from base import BaseMixture
from base import _log_B
import graphics

import pickle
import os
import numpy as np
import scipy.stats
from scipy.special import psi,betaln
from scipy.misc import logsumexp
import h5features
import warnings

class DPVariationalGaussianMixture(BaseMixture):

    """
    Variational Bayesian Estimation of a Gaussian Mixture with Dirichlet Process
    
    This class allows to infer an approximate posterior distribution over the
    parameters of a Gaussian mixture distribution.
    
    The weights distribution is a Dirichlet distribution with parameter _alpha
    (see Bishop's book p474-486)
    
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

    Other parameters
    ----------------                
    
    alpha_0 : float, Optional | defaults to None.
        The prior parameter on the weight distribution (Beta).
        A high value of _alpha_0 will lead to equal weights, while a low value
        will allow some clusters to shrink and disappear. Must be greater than 0.
    
        If it is None, the value is set to 1/n_components                         
    
    beta_0 : float, Optional | defaults to None.
        The precision prior on the mean distribution (Gaussian).
        Must be greater than 0.
    
        If it is None, the value is set to 1.0                         
    
    nu_0 : float, Optional | defaults to None.
        The prior of the number of degrees of freedom on the covariance
        distributions (Wishart). Must be greater or equal to dim.
    
        If it is None, the value is set to dim

    Attributes
    ----------
    
    name : str
        The name of the method : 'VBGMM'
    
    _alpha : array of floats (n_components,2)
        Contains the parameters of the weight distribution (Beta)
    
    _beta : array of floats (n_components,)
        Contains coefficients which are multipied with the precision matrices
        to form the precision matrix on the Gaussian distribution of the means.    
    
    _nu : array of floats (n_components,)
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
    
    _early_stopping : bool
        A boolean to deal with test data (case of the early stopping)
    
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
    ValueError: if the parameters are inconsistent, for example if the
    cluster number is negative, init_type is not in ['resp','mcw']...
    
    References
    ----------
    'Variational Inference for Dirichlet Process Mixtures', D. Blei and M. Jordan
 
    """
    
    def __init__(self, n_components=1,init="VBGMM",alpha_0=None,reg_covar=1e-6,\
                 beta_0=None,nu_0=None,patience=0,type_init='resp'):
        
        super(DPVariationalGaussianMixture, self).__init__()
        
        self.name = 'DPGMM'
        self.n_components = n_components
        self.covariance_type = "full"
        self.init = init
        self.type_init = type_init
        self.reg_covar = reg_covar
        
        self._alpha_0 = alpha_0
        self._beta_0 = beta_0
        self._nu_0 = nu_0
        
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
          
    def _initialize(self,points_data,points_test):
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

        # Prior mean and prior covariance
        self._means_prior = np.mean(points_data,axis=0)
        self._inv_prec_prior = np.cov(points_data.T)
        
        self._check_hyper_parameters(n_points,dim)
        
        if self.type_init == 'resp':
            log_assignements = initial.initialize_log_assignements(self.init,self.n_components,points_data,points_test)
            self._inv_prec = np.empty((self.n_components,dim,dim))
            self._log_det_inv_prec = np.empty(self.n_components)
            self.cov = np.empty((self.n_components,dim,dim))
            self._alpha = np.empty((self.n_components,2))
            self._step_M(points_data,log_assignements)
        
        elif self.type_init == 'mcw':
            #Means, covariances and weights
            means,cov,log_weights = initial.initialize_mcw(self.init,self.n_components,points_data,points_test)
            self.cov = cov
            self.means = means
            self.log_weights = log_weights
            
            # Hyper parameters
            N = np.exp(log_weights)
            self._alpha = np.empty((self.n_components,2))
            for i in range(self.n_components):
                sum_N = np.sum(N[i+1::])
                self._alpha[i] = np.asarray([1+N[i],self._alpha_0+sum_N])
            self._beta = self._beta_0 + N
            self._nu = self._nu_0 + N
            
            # Matrix W
            self._inv_prec = cov * self._nu[:,np.newaxis,np.newaxis]
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
        log_resp = np.zeros((n_points,self.n_components))
        log_weights = np.zeros(self.n_components)
        
        for i in range(self.n_components):
            if i==0:
                log_weights[i] = psi(self._alpha[i][0]) - psi(np.sum(self._alpha[i]))
            else:
                log_weights[i] = psi(self._alpha[i][0]) - psi(np.sum(self._alpha[i]))
                log_weights[i] += log_weights[i-1] + psi(self._alpha[i-1][1]) - psi(self._alpha[i-1][0])
        
        log_gaussian = _log_normal_matrix(points,self.means,self.cov,'full')
        digamma_sum = np.sum(scipy.special.psi(.5 * (self._nu - np.arange(0, dim)[:,np.newaxis])),0)
        log_lambda = digamma_sum + dim * np.log(2) + dim/self._beta
        
        log_prob = log_weights + log_gaussian + 0.5 * (log_lambda - dim * np.log(self._nu))
        
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
        X_barre = np.tile(1/N, (dim,1)).T * np.dot(resp.T,points)        #Array (n_components,dim)
        S = np.zeros((self.n_components,dim,dim))                        #Array (n_components,dim,dim)
        for i in range(self.n_components):
            diff = points - X_barre[i]
            diff_weighted = diff * np.tile(resp[:,i:i+1], (1,dim))
            S[i] = 1/N[i] * np.dot(diff_weighted.T,diff)
            
            S[i] += self.reg_covar * np.eye(dim)
        
        #Parameters update
        for i in range(self.n_components):
            sum_N = np.sum(N[i+1::])
            self._alpha[i] = np.asarray([1+N[i],self._alpha_0+sum_N])
        self._beta = self._beta_0 + N
        self._nu = self._nu_0 + N
        
        means = self._beta_0 * self._means_prior + np.tile(N,(dim,1)).T * X_barre
        self.means = means * np.tile(np.reciprocal(self._beta), (dim,1)).T
        self.means_estimated = self.means
        
        for i in range(self.n_components):
            diff = X_barre[i] - self._means_prior
            product = self._beta_0 * N[i]/self._beta[i] * np.outer(diff,diff)
            self._inv_prec[i] = self._inv_prec_prior + N[i] * S[i] + product
            
            det_inv_prec = np.linalg.det(self._inv_prec[i])
            self._log_det_inv_prec[i] = np.log(det_inv_prec)
            self.cov[i] = self._inv_prec[i] / self._nu[i]
            
        self.log_weights = logsumexp(log_resp, axis=0) - np.log(n_points)
        
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
            
            lower_bound[i] = _log_B(prec_prior,self._nu_0) - _log_B(prec[i],self._nu[i])
            
            resp_i = resp[:,i:i+1]
            log_resp_i = log_resp[:,i:i+1]
            
            lower_bound[i] -= np.sum(resp_i*log_resp_i)
            lower_bound[i] += dim*0.5*(np.log(self._beta_0) - np.log(self._beta[i]))
        
        result = np.sum(lower_bound)
        result -= self.n_components * betaln(1,self._alpha_0)
        result += np.sum(betaln(self._alpha.T[0],self._alpha.T[1]))
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
        N = np.exp(logsumexp(log_resp,axis=0)) + 10*np.finfo(resp.dtype).eps    #Array (n_components,)
        X_barre = np.tile(1/N, (dim,1)).T * np.dot(resp.T,points)               #Array (n_components,dim)
        S = np.zeros((self.n_components,dim,dim))                               #Array (n_components,dim,dim)
        for i in range(self.n_components):
            diff = points - X_barre[i]
            diff_weighted = diff * np.tile(np.sqrt(resp[:,i:i+1]), (1,dim))
            S[i] = 1/N[i] * np.dot(diff_weighted.T,diff_weighted)
            
            S[i] += self.reg_covar * np.eye(dim)
        
        prec = np.linalg.inv(self._inv_prec)
        prec_prior = np.linalg.inv(self._inv_prec_prior)
        
        lower_bound = np.zeros(self.n_components)
        
        for i in range(self.n_components):
            
            digamma_sum = 0
            for j in range(dim):
                digamma_sum += scipy.special.psi((self._nu[i] - j)/2)
            log_det_prec_i = digamma_sum + dim * np.log(2) - self._log_det_inv_prec[i] #/!\ Inverse
            
            #First line
            lower_bound[i] = log_det_prec_i - dim/self._beta[i] - self._nu[i]*np.trace(np.dot(S[i],prec[i]))
            diff = X_barre[i] - self.means[i]
            lower_bound[i] += -self._nu[i]*np.dot(diff,np.dot(prec[i],diff.T))
            lower_bound[i] *= 0.5 * N[i]
            
            #Second line
            lower_bound[i] += _log_B(prec_prior,self._nu_0) - _log_B(prec[i],self._nu[i])
            
            resp_i = resp[:,i:i+1]
            log_resp_i = log_resp[:,i:i+1]
            
            lower_bound[i] -= np.sum(resp_i*log_resp_i)
            lower_bound[i] += 0.5 * (self._nu_0 - self._nu[i]) * log_det_prec_i
            lower_bound[i] += dim*0.5*(np.log(self._beta_0) - np.log(self._beta[i]))
            lower_bound[i] += dim*0.5*(1 - self._beta_0/self._beta[i] + self._nu[i])
            
            #Third line without the last term which is not summed
            diff = self.means[i] - self._means_prior
            lower_bound[i] += -0.5*self._beta_0*self._nu[i]*np.dot(diff,np.dot(prec[i],diff.T))
            lower_bound[i] += -0.5*self._nu[i]*np.trace(np.dot(self._inv_prec_prior,prec[i]))
            
            #Terms with alpha
            lower_bound[i] += (N[i] + 1 - self._alpha[i,0]) * (psi(self._alpha[i,0]) - psi(np.sum(self._alpha[i])))
            lower_bound[i] += (np.sum(N[i+1::]) + self._alpha_0 - self._alpha[i,1]) * (psi(self._alpha[i,1]) - psi(np.sum(self._alpha[i])))
        
        result = np.sum(lower_bound)
        result -= self.n_components * betaln(1,self._alpha_0)
        result += np.sum(betaln(self._alpha.T[0],self._alpha.T[1]))
        result -= n_points * dim * 0.5 * np.log(2*np.pi)
        
        return result
    
    def write(self,group):
        """
        A method creating datasets in a group of an hdf5 file in order to save
        the model
        
        Parameters
        ----------
        group : HDF5 group

        """
        group.create_dataset('means',self.means.shape,dtype='float64')
        group['means'][...] = self.means
        group.create_dataset('cov',self.cov.shape,dtype='float64')
        group['cov'][...] = self.cov
        group.create_dataset('log_weights',self.log_weights.shape,dtype='float64')
        group['log_weights'][...] = self.log_weights
        
        initial_parameters = np.asarray([self._alpha_0,self._beta_0,self._nu_0])
        group.create_dataset('initial parameters',initial_parameters.shape,dtype='float64')
        group['initial parameters'][...] = initial_parameters
        group.create_dataset('means prior',self._means_prior.shape,dtype='float64')
        group['means prior'][...] = self._means_prior
        group.create_dataset('inv prec prior',self._inv_prec_prior.shape,dtype='float64')
        group['inv prec prior'][...] = self._inv_prec_prior
        
        group.create_dataset('alpha',self._alpha.shape,dtype='float64')
        group['alpha'][...] = self._alpha
        group.create_dataset('beta',self._beta.shape,dtype='float64')
        group['beta'][...] = self._beta
        group.create_dataset('nu',self._nu.shape,dtype='float64')
        group['nu'][...] = self._nu
        
    def read_and_init(self,group):
        """
        A method reading a group of an hdf5 file to initialize DPGMM
        
        Parameters
        ----------
        group : HDF5 group

        """
        self.means = np.asarray(group['means'].value)
        self.cov = np.asarray(group['cov'].value)
        self.log_weights = np.asarray(group['log_weights'].value)
        
        initial_parameters = group['initial parameters'].value
        self._alpha_0 = initial_parameters[0]
        self._beta_0 = initial_parameters[1]
        self._nu_0 = initial_parameters[2]
        self._means_prior = np.asarray(group['means prior'].value)
        self._inv_prec_prior = np.asarray(group['inv prec prior'].value)
        n_components = len(self.means)
        if n_components != self.n_components:
            warnings.warn('You are now currently working with %s components.'
                          % n_components)
            self.n_components = n_components
        
        #We want to be able to use data written by VBGMM or GMM
        alpha = np.asarray(group['alpha'])
        if len(alpha) == 1:
            self._alpha = np.empty((self.n_components,2))
            N = alpha - self._alpha_0
            for i in range(self.n_components):
                sum_N = np.sum(N[i+1::])
                self._alpha[i] = np.asarray([1+N[i],self._alpha_0+sum_N])
        else:
            self._alpha = alpha
        self._beta = np.asarray(group['beta'])
        self._nu = np.asarray(group['nu'])
        
        # Matrix W
        self._inv_prec = self.cov * self._nu[:,np.newaxis,np.newaxis]
        self._log_det_inv_prec = np.log(np.linalg.det(self._inv_prec))
        
        self._is_initialized = True
        self.type_init ='user'
        self.init = 'user'
    
if __name__ == '__main__':
    
#    points_data = utils.read("D:/Mines/Cours/Stages/Stage_ENS/Code/data/EMGaussienne.data")
#    points_test = utils.read("D:/Mines/Cours/Stages/Stage_ENS/Code/data/EMGaussienne.test")
#    
#    initializations = ["random","plus","kmeans","GMM"]
#    
#    
        
    k=100
    N=1500
    early_stop = False
    
    path = 'D:/Mines/Cours/Stages/Stage_ENS/Code/data/data.pickle'
    with open(path, 'rb') as fh:
        data = pickle.load(fh)
        
    points = data['BUC']
    if early_stop:
        n_points,_ = points.shape
        idx1 = np.random.randint(0,high=n_points,size=N)
        points_data = points[idx1,:]
        idx2 = np.random.randint(0,high=n_points,size=N)
        points_test = points[idx2,:]
    else:
        points_data = points[:N:]
        points_test = None
    
#    data = h5features.Reader('D:/Mines/Cours/Stages/Stage_ENS/Code/data/mfcc_delta_cmn.features').read()
#    points = np.concatenate(data.features(),axis=0)
#    np.random.shuffle(points)
#    n_points,dim = points.shape
#    
#    points_data = points[:N:]
#    points_test = points[N:2*N:]
    
    init = "GMM"
    directory = os.getcwd() + '/../../Results/DPGMM/' + init
    
#    coeffs = 10.0**np.arange(-5,5)
#    coeffs_nu = 13 + coeffs
    
    print(">>predicting")
    DPGMM = DPVariationalGaussianMixture(k,init,type_init='mcw')
    DPGMM.fit(points_data,points_test=points_test,saving='log',directory=directory)
    print(">>creating graphs")
    graphics.create_graph(DPGMM,points_data,directory,'data')
    if early_stop:
        graphics.create_graph(DPGMM,points_test,directory,'test')
    graphics.create_graph_convergence_criterion(DPGMM,directory,DPGMM.type_init)
    graphics.create_graph_weights(DPGMM,directory,DPGMM.type_init)
    graphics.create_graph_entropy(DPGMM,directory,DPGMM.type_init)
    print()
        