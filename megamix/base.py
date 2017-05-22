# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:13:09 2017

@author: Elina THIBEAU-SUTRE
"""
from abc import abstractmethod
import numpy as np
import scipy.linalg
import os
import warnings
import h5py

def _full_covariance_matrix(points,means,weights,log_assignements,reg_covar):
    """
    Compute the full covariance matrices
    """
    nb_points,dim = points.shape
    n_components = len(means)
    
    covariance = np.zeros((n_components,dim,dim))
    
    for i in range(n_components):
        log_assignements_i = log_assignements[:,i]
        
        # We use the square root of the assignement values because values are very
        # small : this ensure that the matrix will be symmetric
        sqrt_assignements_i = np.exp(0.5*log_assignements_i)
        
        points_centered = points - means[i]
        points_centered_weighted = points_centered * sqrt_assignements_i[:,np.newaxis]
        covariance[i] = np.dot(points_centered_weighted.T,points_centered_weighted)
        covariance[i] = covariance[i] / weights[i]
        
        covariance[i] += reg_covar * np.eye(dim)
    
    return covariance

def _spherical_covariance_matrix(points,means,weights,assignements,reg_covar):
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
        
        points_centered = points - means[i]
        points_centered_weighted = points_centered * assignements_i
        product = points_centered * points_centered_weighted
        covariance[i] = np.sum(product)/sum_assignement
        
        covariance[i] += reg_covar
    
    return covariance / dim

def _compute_precisions_chol(cov,covariance_type):
    
     if covariance_type in 'full':
        n_components, n_features, _ = cov.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(cov):
            try:
                cov_chol = scipy.linalg.cholesky(covariance, lower=True)
            except scipy.linalg.LinAlgError:
                raise ValueError(str(k) + "-th covariance matrix non positive definite")
            precisions_chol[k] = scipy.linalg.solve_triangular(cov_chol,
                                                               np.eye(n_features),
                                                               lower=True).T
     return precisions_chol

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
            
    elif covariance_type == "spherical":
        precisions_chol = np.sqrt(np.reciprocal(cov))
        log_det_chol = dim * np.log(precisions_chol)
        
        log_prob = np.empty((n_points,n_components))
        for k, (mu, prec_chol) in enumerate(zip(means,precisions_chol)):
            y = prec_chol * (points - mu)
            log_prob[:,k] = np.sum(np.square(y), axis=1)
            
    return -.5 * (dim * np.log(2*np.pi) + log_prob) + log_det_chol


class BaseMixture():
    """
    Base class for mixture models.
    This abstract class specifies an interface for other mixture classes and
    provides basic common methods for mixture models.
    """

    def __init__(self, n_components=1,init="GMM",n_iter_max=1000,
                 tol=1e-3,patience=0,type_init='resp'):
        
        super(BaseMixture, self).__init__()

        self.n_components = n_components
        self.init = init
        self.type_init = type_init
        
        self._is_initialized = False
        
    def _check_common_parameters(self):
        
        if self.n_components < 1:
            raise ValueError("The number of components cannot be less than 1")
        else:
            self.n_components = int(self.n_components)
            
                        
        if self.type_init not in ['resp','mcw']:
            raise ValueError("Invalid value for 'type_init': %s "
                             "'type_init' should be in "
                             "['resp','mcw']"
                             % self.type_init)
            
    def _check_hyper_parameters(self,n_points,dim):
        """
        This function tests the hyperparameters of the VBGMM and the DBGMM
        """
        
        #Checking alpha_0
        if self._alpha_0 is None:
            self._alpha_0 = 1/self.n_components
        elif self._alpha_0 < 0:
            raise ValueError("alpha_0 must be positive")
        
        #Checking beta_0
        if self._beta_0 is None:
            self._beta_0 = 1.0
        
        #Checking nu_0
        if self._nu_0 is None:
            self._nu_0 = dim
        
        elif self._nu_0 < dim:
            raise ValueError("nu_0 must be more than the dimension of the"
                             "problem or the gamma function won't be defined")
        
    def _check_points(self,points):
        
        if len(points.shape) == 1:
            points=points.reshape(1,len(points))
            
        elif len(points.shape) != 2:
            raise ValueError('Only 2D or 1D arrays are admitted')

        _,dim_points = points.shape
        _,dim_means = self.means.shape
        if dim_means != dim_points:
            raise ValueError('The points given must have the same '
                             'dimension as the problem : ' + str(dim_means))
        return points
    
    @abstractmethod   
    def _convergence_criterion(self,points,log_resp,log_prob_norm):
        """
        The convergence criterion is different for GMM and VBGMM/DPGMM :
            - in GMM the log likelihood is used
            - in VBGMM/DPGMM the lower bound of the log likelihood is used
        """
        pass
        
    @abstractmethod   
    def _convergence_criterion_simplified(self,points,log_resp,log_prob_norm):
        """
        The convergence criterion is different for GMM and VBGMM/DPGMM :
            - in GMM the log likelihood is used
            - in VBGMM/DPGMM the lower bound of the log likelihood is used
        """
        pass
    
    
    def fit(self,points_data,points_test=None,tol=1e-3,patience=None,
            n_iter_max=100,n_iter_fix=None,directory=None,saving=None,
            init=None):
        """
        The EM algorithm
        @param n_iter_max: int, defaults to 1000
            number of iterations maximum that can be done
                         
        @param tol: float, defaults to 1e-3
             The EM algorithm will stop when the difference between the
             convergence criterion
                         
        @param reg_covar: float, defaults to 1e-6
            In order to avoid null covariances this float is added to the
            diagonal of covariances after their computation
            
        @param points_data: an array (n_points,dim)
        @param points_test: an array (n_points,dim) | Optional
        @param draw_graphs: bool | Optional
        @param directory: str | Optional
        @param saving: str | Optional
            Allows the user to save the model parameters in the directory given
            by the user. Options are ['log','final']
        @return self
        """
        
        if directory==None:
            directory = os.getcwd()
        
        self.early_stopping = points_test is not None
            
        resume_iter = True
        first_iter = True
        log_iter = 0
        iter_patience = 0
        
        if patience is None:
            if self.early_stopping:
                warnings.warn('You are using early stopping with no patience. '
                              'Set the patience parameter to 0 to not see this '
                              'message again')
            patience = 0
            iter_patience = 0

        #Initialization
        if init is None or self._is_initialized==False:
            self._initialize(points_data,points_test)
            self.iter = 0
        
        if saving is not None:
            file = h5py.File(directory + "/" + self.name + ".h5", "w")
            grp = file.create_group('init')
            self.write(grp)
            
        
        while resume_iter:
            #EM algorithm
            log_prob_norm_data,log_resp_data = self._step_E(points_data)
            if self.early_stopping:
                log_prob_norm_test,log_resp_test = self._step_E(points_test)
                
            self._step_M(points_data,log_resp_data)
            
            #Computation of the convergence criterion(s)
            self.convergence_criterion_data.append(self._convergence_criterion_simplified(points_data,log_resp_data,log_prob_norm_data))
            if self.early_stopping:
                self.convergence_criterion_test.append(self._convergence_criterion(points_test,log_resp_test,log_prob_norm_test))
            
            
            #Computation of resume_iter
            if first_iter:
                resume_iter = True
                first_iter = False
                
            elif n_iter_fix is not None:
                resume_iter = self.iter < n_iter_fix
            
            elif self.iter > n_iter_max:
                resume_iter = False
            
            elif self.early_stopping:
                criterion = self.convergence_criterion_test[self.iter] - self.convergence_criterion_test[self.iter-1]
                criterion /= len(points_test)
                if criterion < tol:
                    resume_iter = iter_patience < patience
                    iter_patience += 1
                    
            else:
                criterion = self.convergence_criterion_data[self.iter] - self.convergence_criterion_data[self.iter-1]
                criterion /= len(points_data)
                if criterion < tol:
                    resume_iter = iter_patience < patience
                    iter_patience += 1
            
            
            #Saving the model
            if saving is not None and resume_iter == False:
                grp = file.create_group("final")
                self.write(grp)
                
            elif saving=='log' and self.iter == 2**log_iter:
                grp = file.create_group("log_iter" + str(log_iter))
                self.write(grp)
                log_iter += 1
            
            
            self.iter+=1
        
        if saving is not None:
            file.close()
        
        print("Number of iterations :", self.iter)
    
    def predict_log_resp(self,points):
        """
        This function return the logarithm of the norm of the probability of
        each point and their responsibilities
        
        @param points: a 1D or 2D array of points with the same dimension as
                       as the problem
        @return log_prob_norm: the logarithm of the norm of the probability
                               of each point (n_points,)
                log_resp: the logarithm of the responsibilities (n_points,n_components)
        """
        
        points = self._check_points(points)
        
        if self._is_initialized:
            _,log_resp = self._step_E(points)
            return log_resp
    
        else:
            raise Exception("The model is not initialized")
    
    def score_convergence_criterion(self,points):
        """
        This function return the score of the function, which is the logarithm of
        the likelihood for GMM and the logarithm of the lower bound of the likelihood
        for VBGMM and DPGMM
        
        @param points: a 1D or 2D array of points with the same dimension as
                       as the problem
        @return score: (float)
        """
        points = self._check_points(points)
            
        if self._is_initialized:
            log_prob,log_resp = self._step_E(points)
            score = self._convergence_criterion(points,log_resp,log_prob)
            return score
        
        else:
            raise Exception("The model is not fitted")
    
    @abstractmethod
    def write(self,directory,legend):
        """
        A method which saves the model parameters in order to be reloaded and reused
        
        @param directory: str
            The directory path. The model will be saved there.
        @param legend: str
        """
        pass
    
    
    @abstractmethod
    def read_and_init(self,group):
        """
        A method reading a group of an hdf5 file to initialize DPGMM
        
        @param group: HDF5 group
        """
        pass