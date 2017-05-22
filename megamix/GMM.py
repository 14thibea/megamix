# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:34:50 2017

@author: Elina Thibeau-Sutre
"""

from base import BaseMixture
from base import _log_normal_matrix
from base import _full_covariance_matrix
from base import _spherical_covariance_matrix
import initializations as initial
import graphics

import numpy as np
import os
from scipy.misc import logsumexp
import pickle

class GaussianMixture(BaseMixture):

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
        This method initializes the means, covariances and weights of the model
        """
        
        if self.type_init=='resp':
            log_assignements = initial.initialize_log_assignements(self.init,self.n_components,points_data,points_test,self.covariance_type)
            self._step_M(points_data,log_assignements)
        elif self.type_init=='mcw':
            means,cov,log_weights = initial.initialize_mcw(self.init,self.n_components,points_data,points_test,self.covariance_type)
            self.means = means
            self.cov = cov
            self.log_weights = log_weights
            
        self._is_initialized = True
    
    def _step_E(self,points):
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
      
    def _step_M(self,points,log_assignements):
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
        
    def _convergence_criterion_simplified(self,points,log_resp,log_prob_norm):
        """
        This method returns the log likelihood at the end of the k_means.
        
        @param points: an array of points (n_points,dim)
        @return: log likelihood measurement (float)
        """
        return np.sum(log_prob_norm)
    
    def _convergence_criterion(self,points,log_resp,log_prob_norm):
        """
        This method returns the log likelihood at the end of the k_means.
        
        @param points: an array of points (n_points,dim)
        @return: log likelihood measurement (float)
        """
        return np.sum(log_prob_norm)
                
                
    def write(self,group):
        """
        A method creating datasets in a group of an hdf5 file in order to save
        the model
        
        @param group: HDF5 group
        """
        group.create_dataset('means',self.means.shape,dtype='float64')
        group['means'][...] = self.means
        group.create_dataset('cov',self.cov.shape,dtype='float64')
        group['cov'][...] = self.cov
        group.create_dataset('log_weights',self.log_weights.shape,dtype='float64')
        group['log_weights'][...] = self.log_weights
        
    def read_and_init(self,group):
        """
        A method reading a group of an hdf5 file to initialize DPGMM
        
        @param group: HDF5 group
        """
        self.means = np.asarray(group['means'].value)
        self.cov = np.asarray(group['cov'].value)
        self.log_weights = np.asarray(group['log_weights'].value)
        

if __name__ == '__main__':
    
    path = 'D:/Mines/Cours/Stages/Stage_ENS/Code/data/data.pickle'
    with open(path, 'rb') as fh:
        data = pickle.load(fh)
        
    N=1500
    k=100
    early_stop = True
    
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
    GM = GaussianMixture(k,covariance_type="full",type_init='resp')
    
    print(">>predicting")
    GM.fit(points_data,points_test,patience=0,directory=directory,saving='log')
    print(">>creating graphs")
    graphics.create_graph_convergence_criterion(GM,directory,GM.type_init)
    graphics.create_graph_weights(GM,directory,GM.type_init)
    graphics.create_graph_entropy(GM,directory,GM.type_init)
    print()
        