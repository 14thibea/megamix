# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:34:50 2017

@author: Calixi
"""

from base import BaseMixture
from base import _log_normal_matrix
from base import _full_covariance_matrix
from base import _spherical_covariance_matrix
import Initializations as Init

import numpy as np
import os
from scipy.misc import logsumexp
import pickle

class GaussianMixture(BaseMixture):

    def __init__(self, n_components=1,covariance_type="full",init="kmeans"\
                 ,n_iter_max=100,tol=1e-3,reg_covar=1e-6,patience=0,
                 type_init='resp'):
        
        super(GaussianMixture, self).__init__()

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.init = init
        self.type_init = type_init
        
        self.tol = tol
        self.patience = patience
        self.n_iter_max = n_iter_max
        
        self._check_common_parameters()
        self._check_parameters()
        self.reg_covar = reg_covar

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
        This method initializes the means, covariances and weights of the model
        """
        
        if self.type_init=='resp':
            log_assignements = Init.initialize_log_assignements(self.init,self.n_components,points_data,points_test,self.covariance_type)
            self.step_M(points_data,log_assignements)
        elif self.type_init=='mcw':
            means,cov,log_weights = Init.initialize_mcw(self.init,self.n_components,points_data,points_test,self.covariance_type)
            self.means = means
            self.cov = cov
            self.log_weights = log_weights
    
    def step_E(self,points):
        """
        This method returns the list of the soft assignements of each point to each cluster
        @param log_assignements: an array containing the log of soft assignements of every point (n_points,n_components)
        
        
        @param points: an array of points (n_points,dim)
        @return: log of the soft assignements of every point (n_points,n_components)
        """
        log_normal_matrix = _log_normal_matrix(points,self.means,self.cov,self.covariance_type)
        log_product = log_normal_matrix + self.log_weights[:,np.newaxis].T
        log_prob_norm = logsumexp(log_product,axis=1)
        
        log_resp = log_product - log_prob_norm[:,np.newaxis]
        
        return log_prob_norm,log_resp
      
    def step_M(self,points,log_assignements):
        """
        This method computes the new position of each mean and each covariance matrix
        
        @param points: an array of points (n_points,dim)
        @param log_assignements: an array containing the log of soft assignements of every point (n_points,n_components)
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
        
    def convergence_criterion_simplified(self,points,log_resp,log_prob_norm):
        """
        This method returns the log likelihood at the end of the k_means.
        
        @param points: an array of points (n_points,dim)
        @return: log likelihood measurement (float)
        """
        return np.sum(log_prob_norm)
    
    def convergence_criterion(self,points,log_resp,log_prob_norm):
        """
        This method returns the log likelihood at the end of the k_means.
        
        @param points: an array of points (n_points,dim)
        @return: log likelihood measurement (float)
        """
        return np.sum(log_prob_norm)
        

    def set_parameters(self,means=None,cov=None,log_weights=None):
        """
        This method allows the user to change one or more parameters used by the algorithm
        
        @param means: the new means of the clusters     (n_components,dim)
        @param cov: the new covariance matrices         (n_components,dim,dim)
        @param log_weights: the logarithm of the weights(n_components,)
        """
        
        if not means is None:
            n_components,dim = means.shape
            self.means = np.zeros((self.n_components,dim))
            if n_components != self.n_components:
                print("Warning : you decided to work with", self.n_components,
                      "components, the means given are going to be truncated "
                      "or multiplied")
                if n_components < self.n_components:
                    rest = self.n_components - n_components
                    self.means[:n_components:] = means
                    self.means[n_components::] = np.tile(means[-1], (rest,1))
                else:
                    self.means = means[:self.n_components:]
            else:
                self.means = means
                
if __name__ == '__main__':
    
    path = 'D:/Mines/Cours/Stages/Stage_ENS/Code/data/data.pickle'
    with open(path, 'rb') as fh:
        data = pickle.load(fh)
        
    N=1500
    k=100
    early_stop = False
    
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


    init = 'kmeans'
    directory = os.getcwd() + '/../Results/GMM/' + init

    #GMM
#    for i in range(10):
    i=0
    print(i)
    GM = GaussianMixture(k,covariance_type="full",patience=0,tol=1e-3,type_init='resp')
    
    print(">>predicting")
    GM.fit(points_data,points_test)
    print(">>creating graphs")
    GM.create_graph_convergence_criterion(directory,GM.type_init)
    GM.create_graph_weights(directory,GM.type_init)
    GM.create_graph_entropy(directory,GM.type_init)
    print()
        