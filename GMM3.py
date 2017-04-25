# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:34:50 2017

@author: Calixi
"""

import utils
from base import BaseMixture
from base import _compute_precisions_chol
import Initializations as Init

import numpy as np
import os
from scipy.misc import logsumexp
import pickle

def _log_normal_matrix(points,means,cov,covariance_type):
    """
    This method computes the log of the density of probability of a normal law centered. Each line
    corresponds to a point from points.
    
    @param points: an array of points (n_points,dim)
    @param means: an array of k points which are the means of the clusters (n_components,dim)
    @param cov: an array of k arrays which are the covariance matrices (n_components,dim,dim)
    @return: an array containing the log of density of probability of a normal law centered (n_points,n_components)
    """
    n_points,dim = points.shape
    n_components,_ = means.shape
    
    
    if covariance_type == "full":
        precisions_chol = _compute_precisions_chol(cov,covariance_type)
        log_det_chol = np.log(np.linalg.det(precisions_chol))
        
        log_prob = np.empty((n_points,n_components))
        for k, (mu, prec_chol) in enumerate(zip(means,precisions_chol)):
            y = np.dot(points,prec_chol) - np.dot(mu,prec_chol)
            log_prob[:,k] = np.sum(np.square(y), axis=1)
            
    if covariance_type == "spherical":
        precisions_chol = np.sqrt(np.reciprocal(cov))
        log_det_chol = dim * np.log(precisions_chol)
        
        log_prob = np.empty((n_points,n_components))
        for k, (mu, prec_chol) in enumerate(zip(means,precisions_chol)):
            y = prec_chol * (points - mu)
            log_prob[:,k] = np.sum(np.square(y), axis=1)
            
    return -.5 * (n_points * np.log(2*np.pi) + log_prob) + log_det_chol

def _full_covariance_matrix(points,means,log_assignements,reg_covar):
    """
    Compute the full covariance matrices
    """
    nb_points,dim = points.shape
    n_components = len(means)
    
    covariance = np.zeros((n_components,dim,dim))
    
    for i in range(n_components):
        log_assignements_i = log_assignements[:,i:i+1]
        
        # We use the square root of the assignement values because values are very
        # small : this ensure that the matrix will be symmetric
        sqrt_assignements_i = np.tile(np.exp(0.5*log_assignements_i), (1,dim))
        sum_assignement = np.exp(logsumexp(log_assignements_i)) 
        sum_assignement += 10 * np.finfo(log_assignements.dtype).eps
        
        points_centered = points - means[i]
        points_centered_weighted = points_centered * sqrt_assignements_i
        covariance[i] = np.dot(points_centered_weighted.T,points_centered_weighted)
        covariance[i] = covariance[i] / sum_assignement
        
        covariance[i] += reg_covar * np.eye(dim)
    
    return covariance

def _spherical_covariance_matrix(points,means,assignements,reg_covar):
    """
    Compute the coefficients for the spherical covariances matrices
    """
    n_points,dim = points.shape
    n_components = len(means)
    
    covariance = np.zeros(n_components)

    for i in range(n_components):
        assignements_i = assignements[:,i:i+1]
        sum_assignement = np.sum(assignements_i)
        sum_assignement += 10 * np.finfo(assignements.dtype).eps
        
        assignements_duplicated = np.tile(assignements_i, (1,dim))
        points_centered = points - means[i]
        points_centered_weighted = points_centered * assignements_duplicated
        product = np.dot(points_centered_weighted,points_centered.T)
        covariance[i] = np.trace(product)/sum_assignement
        
        covariance[i] += reg_covar
    
    return covariance / dim

class GaussianMixture(BaseMixture):

    def __init__(self, n_components=1,covariance_type="full",init="kmeans"\
                 ,n_iter_max=100,tol=1e-3,reg_covar=1e-6,patience=0):
        
        super(GaussianMixture, self).__init__()

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.init = init
        
        self.tol = tol
        self.patience = patience
        self.n_iter_max = n_iter_max
        
        self._check_common_parameters()
        self._check_parameters()
        self.reg_covar = reg_covar

    def _check_parameters(self):
        
        if self.init not in ['random', 'plus', 'kmeans']:
            raise ValueError("Invalid value for 'init': %s "
                             "'init' should be in "
                             "['random', 'plus', 'kmeans']"
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
        
        n_points = len(points)
        
        log_normal_matrix = _log_normal_matrix(points,self.means,self.cov,self.covariance_type)
        log_weights_duplicated = np.tile(self.log_weights, (n_points,1))
        log_product = log_normal_matrix + log_weights_duplicated
        log_product_sum = np.tile(logsumexp(log_product,axis=1),(self.n_components,1))
        
        return log_product - log_product_sum.T
      
    def step_M(self,points,log_assignements):
        """
        This method computes the new position of each mean and each covariance matrix
        
        @param points: an array of points (n_points,dim)
        @param log_assignements: an array containing the log of soft assignements of every point (n_points,n_components)
        """
        n_points,dim = points.shape
        
        assignements = np.exp(log_assignements)
        
        #Phase 1:
        result_inter = np.dot(assignements.T,points)
        sum_assignements = np.sum(assignements,axis=0)
        sum_assignements_final = np.reciprocal(np.tile(sum_assignements, (dim,1)).T)
        
        self.means_pre = np.copy(self.means)
        self.means = result_inter * sum_assignements_final
        
        #Phase 2:
        if self.covariance_type=="full":
            self.cov_pre = np.copy(self.cov)
            self.cov = _full_covariance_matrix(points,self.means,log_assignements,self.reg_covar)
        elif self.covariance_type=="spherical":
            self.cov = _spherical_covariance_matrix(points,self.means,assignements,self.reg_covar)
                        
        #Phase 3:
        self.log_weights_pre = np.copy(self.log_weights)
        self.log_weights = logsumexp(log_assignements, axis=0) - np.log(n_points)
    
    def convergence_criterion(self,points,log_resp):
        """
        This method returns the log likelihood at the end of the k_means.
        
        @param points: an array of points (n_points,dim)
        @return: log likelihood measurement (float)
        """
        n_points = len(points)
        
        log_normal_matrix = _log_normal_matrix(points,self.means,self.cov,self.covariance_type)
        log_weights_duplicated = np.tile(self.log_weights, (n_points,1))
        log_product = log_normal_matrix + log_weights_duplicated
        log_product = logsumexp(log_product,axis=1)
        return np.sum(log_product)
        
    def create_path(self):
        """
        Create a directory to store the graphs
        
        @return: the path of the directory (str)
        """
        dir_path = 'GMM/' + self.covariance_type + '/'
        directory = os.path.dirname(dir_path)
    
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)  
        return directory

    
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
    
    #Lecture du fichier
    points_data = utils.read("D:/Mines/Cours/Stages/Stage_ENS/Code/data/EMGaussienne.data")
    points_test = utils.read("D:/Mines/Cours/Stages/Stage_ENS/Code/data/EMGaussienne.test")
    
    path = 'D:/Mines/Cours/Stages/Stage_ENS/Code/data/data.pickle'
    with open(path, 'rb') as fh:
        data = pickle.load(fh)
        
    N=1500
        
    points = data['BUC']
    points_data = points[:N:]
    idx = np.random.randint(0,high=len(points),size=N)
    points_test = points[idx,:]

#    cluster_number = np.arange(2,10) * 10
    
    #GMM
#    for i in cluster_number:
    i=100
    print(i)
    GMM = GaussianMixture(i,covariance_type="full",patience=0,tol=1e-5,reg_covar=1e-6)
    
    print(">>predicting")
#    log_assignements_data,log_assignements_test = GMM.predict_log_assignements(points_data,points_test)
    log_assignements_data = GMM.predict_log_assignements(points_data,draw_graphs=False)
    print(">>creating graphs")
#    GMM.create_graph(points_data,log_assignements_data,str(i) + "_data")
#    GMM.create_graph(points_test,log_assignements_test,str(i) + "_test")
    GMM.create_graph_convergence_criterion(i)
    GMM.create_graph_weights(i)
#    GMM.create_graph_entropy(i)
#    GMM.create_graph_MDS(i)
    print()