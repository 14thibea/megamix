# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:37:08 2017

@author: Calixi
"""

import utils
import Initializations as Init
from base import BaseMixture
from base import _log_normal_matrix
from base import _full_covariance_matrix
from base import _spherical_covariance_matrix

import pickle
import os
import numpy as np
import scipy.special
from scipy.misc import logsumexp
#import sklearn_test

class VariationalGaussianMixture(BaseMixture):
    """
    Variational Bayesian Estimation of a Gaussian Mixture
    
    This class allows to infer an approximate posterior distribution over the
    parameters of a Gaussian mixture distribution.
    
    The weights distribution is a Dirichlet distribution with parameter _alpha
    (see Bishop's book p474-486)

        
    The hyperparameters alpha_0, beta_0, nu_0 may be initialized by
    _check_hyper_parameters() in base.py if not initialized by the user
    
    Parameters :
    ---------------
    @param n_components: int, defaults to 1
                         number of clusters used

    @param init:         str, defaults to 'kmeans'
                         method used in order to perform the initialization
                         must be in ['random','plus','AF_KMC','kmeans','GMM']
                         
    @param n_iter_max:   int, defaults to 1000
                         number of iterations maximum that can be done
                         
    @param tol:          float, defaults to 1e-3
                         The EM algorithm will stop when the difference between the
                         convergence criterion
                         
    @param reg_covar:    float, defaults to 1e-6
                         In order to avoid null covariances this float is added to the
                         diagonal of covariances after their computation
                         
    @param _alpha_0:     float, optional defaults to None
                         The prior paramete on the weight distribution (Dirichlet).
                         A high value of _alpha_0 will lead to equal weights, while
                         a low value will allow some clusters to shrink and disappear.
                         Must be greater than 0.
                         
                         If it is None, the value is set to 1/n_components
                         
    @param _beta_0:      float, optional defaults to None
                         The precision prior on the mean distribution (Gaussian).
                         Must be greater than 0.
                         
                         If it is None, the value is set to 1.0
                         
    @param _nu_0:        float, optional defaults to None
                         The prior of the number of degrees of freedom on the covariance
                         distributions (Wishart).
                         Must be greater or equal to dim.
                         
                         If it is None, the value is set to dim
                         
    @param patience:     int, optional defaults to 0
                         Allows the user to resume the computation of the EM algorithm
                         after that the convergence criterion was reached
        
    Attributes :
    ---------------
    @attribute _alpha:
    @attribute _beta:
    @attribute _nu:
    @attribute inv_prec:
    @attribute log_det_inv_prec:
    @attribute cov
    @attribute means
    @attribute log_weights:
    @attribute cov_estimated:
    @attribute means_estimated:
        
    (inherited from BaseMixture)
    @attribute iter:
    @attribute test_exists:
    @attribute convergence_criterion_data:
    @attribute convergence_criterion_test:
    @attribute 
    
    The hyperparameters alpha_0, beta_0, nu_0 may be initialized by
    _check_hyper_parameters() in base.py if not initialized by the user
    """
    
    
    

    def __init__(self, n_components=1,init="GMM",n_iter_max=1000,alpha_0=None,\
                 beta_0=None,nu_0=None,tol=1e-3,reg_covar=1e-6,patience=0, \
                 type_init='resp'):
        
        super(VariationalGaussianMixture, self).__init__()

        self.n_components = n_components
        self.covariance_type = "full"
        self.init = init
        self.type_init = type_init
        self.n_iter_max = n_iter_max
        self.tol = tol
        self.patience = patience
        self.reg_covar = reg_covar
        
        self._alpha_0 = alpha_0
        self._beta_0 = beta_0
        self._nu_0 = nu_0
        
        self._check_common_parameters()
        self._check_parameters()
        
    def _check_parameters(self):
        
        if self.init not in ['random', 'plus', 'kmeans', 'AF_KMC', 'GMM']:
            raise ValueError("Invalid value for 'init': %s "
                             "'init' should be in "
                             "['random', 'plus', 'kmeans','AF_KMC','GMM']"
                             % self.init)
            
    def _sampling_Normal_Wishart(self):
        """
        Sampling mu and sigma from Normal-Wishart distribution.

        """
        # Create the matrix A of the Bartlett decomposition (cf Wikipedia)
        n_components,dim,_ = self.inv_prec.shape

        for i in range(self.n_components):
            prec = np.linalg.inv(self.inv_prec[i])
            chol = np.linalg.cholesky(prec)
            
            A_diag = np.sqrt(np.random.chisquare(self._nu[i] - np.arange(0,dim), size = dim))
            A = np.diag(A_diag)
            A[np.tri(dim,k=-1,dtype = bool)] = np.random.normal(size = (dim*(dim-1))//2)
            
            X = np.dot(chol,A)
            prec_estimated = np.dot(X,X.T)
            
            self.cov_estimated[i] = np.linalg.inv(prec_estimated)
        
            self.means_estimated[i] = np.random.multivariate_normal(self.means[i] , self.cov_estimated[i]/self._beta[i])
    
    def _initialize(self,points_data,points_test=None):
        """
        This method initializes the Variational Gaussian Mixture by setting the values
        of the means, the covariances and other parameters specific (alpha, beta, nu)
        @param points: an array (n_points,dim)
        @param alpha_0: a float which influences on the number of cluster kept
        @param beta_0: a float
        @param nu_0: a float
        """
        
        n_points,dim = points_data.shape
        
        #Prior mean and prior W-1
        self.means_prior = np.mean(points_data,axis=0)
        if self.covariance_type == 'full':
            self.inv_prec_prior = np.cov(points_data.T)
        elif self.covariance_type == 'spherical':
            self.inv_prec_prior = np.var(points_data)
        
        self._check_hyper_parameters(n_points,dim)
        
        if self.type_init=='resp':
            log_assignements = Init.initialize_log_assignements(self.init,self.n_components,points_data,points_test)
            
            self.inv_prec = np.empty((self.n_components,dim,dim))
            self.log_det_inv_prec = np.empty(self.n_components)
            self.cov = np.empty((self.n_components,dim,dim))
            self.step_M(points_data,log_assignements)
            
        elif self.type_init=='mcw':
            # Means, covariances and weights
            means,cov,log_weights = Init.initialize_mcw(self.init,self.n_components,points_data)
            self.cov = cov
            self.means = means
            self.log_weights = log_weights
            
            # Hyperparametres
            N = np.exp(log_weights)
            self._alpha = self._alpha_0 + N
            self._beta = self._beta_0 + N
            self._nu = self._nu_0 + N
            
            # Matrix W
            self.inv_prec = cov * self._nu[:,np.newaxis,np.newaxis]
            self.log_det_inv_prec = np.log(np.linalg.det(self.inv_prec))
            
        # In case of using _sampling_Normal_Wishart()
        self.cov_estimated = self.cov
        self.means_estimated = self.means
    
    def step_E(self, points):
        """
        In this step the algorithm evaluates the responsibilities of each points in each cluster
        
        @param points: an array                                     (n_points,dim)
        @return log_resp: an array containing the logarithm of the 
                          responsibilities                          (n_points,n_components)
                log_prob_norm: logarithm of the probability of each
                               sample in points                     (n_points,)
        """
        
        n_points,dim = points.shape
        log_prob = np.zeros((n_points,self.n_components))
        
        log_weights = scipy.special.psi(self._alpha) - scipy.special.psi(np.sum(self._alpha))       
        log_gaussian = _log_normal_matrix(points,self.means,self.cov,'full')
        digamma_sum = np.sum(scipy.special.psi(.5 * (self._nu - np.arange(0, dim)[:,np.newaxis])),0)
        log_lambda = digamma_sum + dim * np.log(2) + dim/self._beta
        
        log_prob = log_weights + log_gaussian + 0.5 * (log_lambda - dim * np.log(self._nu))
        
        log_prob_norm = logsumexp(log_prob, axis=1)
        log_resp = log_prob - log_prob_norm[:,np.newaxis]
                    
        return log_prob_norm,log_resp
    
    def _estimate_wishart_full(self,N,X_barre,S):
        """
        This method computes the new value of inv_prec with given parameteres
        (in the case of full covariances)
        
        @param N: the empirical weights     (n_components,)
        @param X_barre: the empirical means (n_components,dim)
        @param S: the empirical covariances (n_components,dim,dim)
        """
        for i in range(self.n_components):
            diff = X_barre[i] - self.means_prior
            product = self._beta_0 * N[i]/self._beta[i] * np.outer(diff,diff)
            self.inv_prec[i] = self.inv_prec_prior + N[i] * S[i] + product
            
    def _estimate_wishart_spherical(self,N,X_barre,S):
        """
        This method computes the new value of inv_prec with given parameteres
        (in the case of spherical covariances)
        
        @param N: the empirical weights     (n_components)
        @param X_barre: the empirical means (n_components,dim)
        @param S: the empirical covariances (n_components,)
        """
        for i in range(self.n_components):
            diff = X_barre[i] - self.means_prior
            product = self._beta_0 * N[i] / self._beta[i] * np.mean(np.square(diff), 1)
            self.inv_prec[i] = self.inv_prec_prior + N[i] * S[i] + product
        # To test
                
    def step_M(self,points,log_resp):
        """
        In this step the algorithm updates the values of the parameters (means, covariances,
        alpha, beta, nu).
        
        @param points: an array of points                       (n_points,dim)
        @param log_resp: an array containing the logarithm of
                         the responsibilities                   (n_points,n_components)
        """
        
        n_points,dim = points.shape
        
        resp = np.exp(log_resp)
        
        # Convenient statistics
        N = np.sum(resp,axis=0) + 10 * np.finfo(resp.dtype).eps                    #Array (n_components,)
        X_barre = np.dot(resp.T,points) / N[:,np.newaxis]                          #Array (n_components,dim)
        if self.covariance_type=='full':
            S = _full_covariance_matrix(points,X_barre,N,log_resp,self.reg_covar)  #Array (n_components,dim,dim)
        elif self.covariance_type=='spherical':
            S = _spherical_covariance_matrix(points,X_barre,N,resp,self.reg_covar) #Array (n_components,)
        
        #Parameters update
        self._alpha = self._alpha_0 + N
        self._beta = self._beta_0 + N
        self._nu = self._nu_0 + N
        
        self.means = (self._beta_0 * self.means_prior + N[:, np.newaxis] * X_barre) / self._beta[:, np.newaxis]
        
        if self.covariance_type=="full":
            self._estimate_wishart_full(N,X_barre,S)
            det_inv_prec = np.linalg.det(self.inv_prec)
            self.log_det_inv_prec = np.log(det_inv_prec)
            self.cov = self.inv_prec / self._nu[:,np.newaxis,np.newaxis]
            
        elif self.covariance_type=="spherical":
            self._estimate_wishart_spherical(N,X_barre,S)
            det_inv_prec = self.inv_prec**dim
            self.log_det_inv_prec = np.log(det_inv_prec)
            self.cov = self.inv_prec / self._nu
        
        self.log_weights = logsumexp(log_resp, axis=0) - np.log(n_points)
        
    def convergence_criterion(self,points,log_resp,log_prob_norm):
        """
        Compute the lower bound of the likelihood using the Bishop's book formula.
        The formula cannot be simplified (as it is done in scikit-learn) as we also
        use it to calculate the lower bound of test points, in this case no
        simplification can be done.
        
        @param points: an array of points (n_points,dim)
        @param log resp: the logarithm of the soft assignements of each point to
                         each cluster     (n_points,n_components)
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
        
        prec = np.linalg.inv(self.inv_prec)
        prec_prior = np.linalg.inv(self.inv_prec_prior)
        
        lower_bound = np.zeros(self.n_components)
        
        for i in range(self.n_components):
            
            log_weights_i = scipy.special.psi(self._alpha[i]) - scipy.special.psi(np.sum(self._alpha))
            
            digamma_sum = 0
            for j in range(dim):
                digamma_sum += scipy.special.psi((self._nu[i] - j)/2)
            log_det_prec_i = digamma_sum + dim * np.log(2) - self.log_det_inv_prec[i] #/!\ Inverse
            
            #First line
            lower_bound[i] = log_det_prec_i - dim/self._beta[i] - self._nu[i]*np.trace(np.dot(S[i],prec[i]))
            diff = X_barre[i] - self.means[i]
            lower_bound[i] += -self._nu[i]*np.dot(diff,np.dot(prec[i],diff.T))
            lower_bound[i] *= 0.5 * N[i]
            
            #Second line
            lower_bound[i] += (self._alpha_0 - self._alpha[i]) * log_weights_i
            lower_bound[i] += utils.log_B(prec_prior,self._nu_0) - utils.log_B(prec[i],self._nu[i])
            
            resp_i = resp[:,i:i+1]
            log_resp_i = log_resp[:,i:i+1]
            
            lower_bound[i] += np.sum(resp_i) * log_weights_i - np.sum(resp_i*log_resp_i)
            lower_bound[i] += 0.5 * (self._nu_0 - self._nu[i]) * log_det_prec_i
            lower_bound[i] += dim*0.5*(np.log(self._beta_0) - np.log(self._beta[i]))
            lower_bound[i] += dim*0.5*(1 - self._beta_0/self._beta[i] + self._nu[i])
            
            #Third line without the last term which is not summed
            diff = self.means[i] - self.means_prior
            lower_bound[i] += -0.5*self._beta_0*self._nu[i]*np.dot(diff,np.dot(prec[i],diff.T))
            lower_bound[i] += -0.5*self._nu[i]*np.trace(np.dot(self.inv_prec_prior,prec[i]))
                
        result = np.sum(lower_bound)
        result += utils.log_C(self._alpha_0 * np.ones(self.n_components))- utils.log_C(self._alpha)
        result -= n_points * dim * 0.5 * np.log(2*np.pi)
        
        return result
        
    def create_path(self):
        """
        Create a directory to store the graphs
        
        @return: the path of the directory (str)
        """
        dir_path = 'VBGMM/' + self.init + '/'
        directory = os.path.dirname(dir_path)
    
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)  
        return directory
    
if __name__ == '__main__':
    
    points_data = utils.read("D:/Mines/Cours/Stages/Stage_ENS/Code/data/EMGaussienne.data")
    points_test = utils.read("D:/Mines/Cours/Stages/Stage_ENS/Code/data/EMGaussienne.test")
    
    initializations = ["random","plus","kmeans","GMM"]
    
    path = 'D:/Mines/Cours/Stages/Stage_ENS/Code/data/data.pickle'
    with open(path, 'rb') as fh:
        data = pickle.load(fh)
    
    k=100
    N=10000
        
    points = data['BUC']
    n_points,dim = points.shape
    idx = np.random.randint(0,high=n_points,size=N)
    points_data = points[idx,:]
    idx = np.random.randint(0,high=n_points,size=N)
    points_test = points[idx,:]
#    points_data = points[:N:]
    
    
#    for init in initializations:
#        print()
#        print(init)
#        print()
    init="GMM"

#    for j in np.arange(2,11):
    j=0
    print(j)
    print(">>predicting")
    VBGMM = VariationalGaussianMixture(k,init,tol=1e-3,patience=0,type_init='mcw')
    log_resp_data,log_resp_test = VBGMM.predict_log_assignements(points_data,points_test)
#    log_resp_data_ = VBGMM.predict_log_assignements(points_data,draw_graphs=False)
    print(">>creating graphs")
    VBGMM.create_graph_convergence_criterion(VBGMM.type_init)
    VBGMM.create_graph_weights(VBGMM.type_init)
#    VBGMM.create_graph_MDS(j)
#    VBGMM.create_graph_entropy(j)
#    print()