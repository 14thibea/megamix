# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:37:08 2017

@author: Calixi
"""

import utils
import Initializations as Init
from base import BaseMixture
from base import _log_normal_matrix

import pickle
import os
import numpy as np
import scipy.special
from scipy.misc import logsumexp
#import sklearn_test

class VariationalGaussianMixture(BaseMixture):
    
    """
    
    
    
    
    The hyperparameters alpha_0, beta_0, nu_0 may be initialized by
    _check_hyper_parameters() in base.py if not initialized by the user
    
    """

    def __init__(self, n_components=1,init="GMM",n_iter_max=1000,alpha_0=None,\
                 beta_0=None,nu_0=None,tol=1e-3,reg_covar=1e-6,patience=0):
        
        super(VariationalGaussianMixture, self).__init__()

        self.n_components = n_components
        self.covariance_type = "full"
        self.init = init
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
        
        if self.init not in ['random', 'plus', 'kmeans','GMM']:
            raise ValueError("Invalid value for 'init': %s "
                             "'init' should be in "
                             "['random', 'plus', 'kmeans','GMM']"
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
        
        self.means_init = np.mean(points_data,axis=0)
        self.inv_prec_init = Init.initialization_full_covariances(self.n_components,points_data)# * self._nu_0
        
        means,cov,log_weights = Init.initialize_mcw(self.init,self.n_components,points_data)
        self.cov = cov
        self.means = means
        self.log_weights = log_weights
        
        
        self.cov_estimated = cov
        self.means_estimated = means
        
        
        self._check_hyper_parameters(n_points,dim)
        N = np.exp(log_weights)
        self._alpha = self._alpha_0 + N
        self._beta = self._beta_0 + N
        self._nu = self._nu_0 + N
        
        self.inv_prec = cov * self._nu[:,np.newaxis,np.newaxis]
        self.log_det_inv_prec = np.log(np.linalg.det(self.inv_prec))
    
    def step_E(self, points):
        """
        In this step the algorithm evaluates the responsibilities of each points in each cluster
        
        @param points: an array (n_points,dim)
        @return resp: an array containing the responsibilities (n_points,n_components)
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
    
    def step_M(self,points,log_resp):
        """
        In this step the algorithm updates the values of the parameters (means, covariances,
        alpha, beta, nu).
        
        @param points: an array (n_points,dim)
        @param resp: an array containing the responsibilities (n_points,n_components)
        """
        
        n_points,dim = points.shape
        
        resp = np.exp(log_resp)
        
        # Convenient statistics
        N = np.sum(resp,axis=0) + 10 * np.finfo(resp.dtype).eps                 #Array (n_components,)
        X_barre = np.dot(resp.T,points) / N[:,np.newaxis]                       #Array (n_components,dim)
        S = np.zeros((self.n_components,dim,dim))                               #Array (n_components,dim,dim)
        for i in range(self.n_components):
            diff = points - X_barre[i]
            S[i] = np.dot(resp[:,i] * diff.T,diff) / N[i]
            
            S[i] += self.reg_covar * np.eye(dim)
        
        #Parameters update
        self._alpha = self._alpha_0 + N
        self._beta = self._beta_0 + N
        self._nu = self._nu_0 + N
        
        self.means = (self._beta_0 * self.means_init + N[:, np.newaxis] * X_barre) / self._beta[:, np.newaxis]
        
        for i in range(self.n_components):
            diff = X_barre[i] - self.means_init
            product = self._beta_0 * N[i]/self._beta[i] * np.outer(diff,diff)
            self.inv_prec[i] = self.inv_prec_init + N[i] * S[i] + product
            
            det_inv_prec = np.linalg.det(self.inv_prec[i])
            self.log_det_inv_prec[i] = np.log(det_inv_prec)
            self.cov[i] = self.inv_prec[i] / self._nu[i]
            
        self.log_weights = logsumexp(log_resp, axis=0) - np.log(n_points)
    
    def convergence_criterion(self,points,log_resp,log_prob_norm):
        
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
        prec_init = np.linalg.inv(self.inv_prec_init)
        
        log_weights = np.zeros(self.n_components)
        log_det_prec = np.zeros(self.n_components)
        lower_bound = np.zeros(self.n_components)
        
        for i in range(self.n_components):
            
            log_weights[i] = scipy.special.psi(self._alpha[i]) - scipy.special.psi(np.sum(self._alpha))
            
            digamma_sum = 0
            for j in range(dim):
                digamma_sum += scipy.special.psi((self._nu[i] - j)/2)
            log_det_prec[i] = digamma_sum + dim * np.log(2) - self.log_det_inv_prec[i] #/!\ Inverse
            
            #First line
            lower_bound[i] = log_det_prec[i] - dim/self._beta[i] - self._nu[i]*np.trace(np.dot(S[i],prec[i]))
            diff = X_barre[i] - self.means[i]
            lower_bound[i] += -self._nu[i]*np.dot(diff,np.dot(prec[i],diff.T)) - dim*np.log(2*np.pi)
            lower_bound[i] *= 0.5 * N[i]
            
            #Second line
            lower_bound[i] += (self._alpha_0 - self._alpha[i]) * log_weights[i]
            lower_bound[i] += utils.log_B(prec_init,self._nu_0) - utils.log_B(prec[i],self._nu[i])
            
            resp_i = resp[:,i:i+1]
            log_resp_i = log_resp[:,i:i+1]
            
            lower_bound[i] += np.sum(resp_i) * log_weights[i] - np.sum(resp_i*log_resp_i)
            lower_bound[i] += 0.5 * (self._nu_0 - self._nu[i]) * log_det_prec[i]
            lower_bound[i] += dim*0.5*(np.log(self._beta_0) - np.log(self._beta[i]))
            lower_bound[i] += dim*0.5*(1 - self._beta_0/self._beta[i] + self._nu[i])
            
            #Third line without the last term which is not summed
            diff = self.means[i] - self.means_init
            lower_bound[i] += -0.5*self._beta_0*self._nu[i]*np.dot(diff,np.dot(prec[i],diff.T))
            lower_bound[i] += -0.5*self._nu[i]*np.trace(np.dot(self.inv_prec_init,prec[i]))
                
        result = np.sum(lower_bound)
        result += utils.log_C(self._alpha_0 * np.ones(self.n_components))- utils.log_C(self._alpha)
        
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
    N=1500
        
    points = data['BUC']
    points_data = points[:N:]
    idx = np.random.randint(0,high=len(points),size=N)
    points_test = points[idx,:]
    
    _,dim = points_data.shape
    
    
#    for init in initializations:
#        print()
#        print(init)
#        print()
    init="GMM"

#    for j in np.arange(2,11):
    j=0
    print(j)
    print(">>predicting")
    VBGMM = VariationalGaussianMixture(k,init,tol=1e-4,patience=0)
    log_resp_data,log_resp_test = VBGMM.predict_log_assignements(points_data,points_test)
#    log_resp_data = VBGMM.predict_log_assignements(points_data,draw_graphs=False)
    print(">>creating graphs")
#    VBGMM.create_graph_convergence_criterion(j)
#    VBGMM.create_graph_weights(j)
#    VBGMM.create_graph_MDS(j)
#    VBGMM.create_graph_entropy(j)
#    print()
    
    utils.write('VBGMM/test.csv',log_resp_data)