# -*- coding: utf-8 -*-
#
#Created on Mon Apr 10 11:34:50 2017
#
#author: Elina Thibeau-Sutre
#
#cython: profile=True

from base_cython cimport BaseMixture
from base_cython cimport logsumexp_axis,cholupdate
from base_cython cimport _log_normal_matrix
from megamix.batch.initializations import initialization_plus_plus

import numpy as np
import scipy
import cython
from scipy.linalg.cython_lapack cimport dpotrf
from cython.view cimport array as cvarray
from libc.math cimport log,sqrt
from basic_operations cimport soustract2Dby2D_idx, multiply3Dbyscalar, multiply2Dby2D_idx
from basic_operations cimport multiply3Dbyvect2D, multiply2Dbyvect2D, multiply2Dbyscalar
from basic_operations cimport divide3Dbyvect2D,divide2Dbyvect2D,divide3Dbyscalar,divide2Dbyscalar
from basic_operations cimport transpose_spe_f2c_and_write, dot_spe_c
from basic_operations cimport add2Dand2D,add2Dscalar, soustract2Dby2D
from basic_operations cimport initialize,erase_above_diag,exp2D,sqrt2D,log2D
from basic_operations cimport cast3Din2D,cast2Din3D,update2D,update3D,transpose_spe_f2c
from basic_operations cimport add2Dscalar_reduce,sum2D

cdef class GaussianMixture(BaseMixture):
    
    cdef double kappa
    cdef int n_jobs #Not used yet
    cdef int update

                
    def __init__(self, int n_components=1,double kappa=1.0,
                 double reg_covar=1e-6,int n_jobs=1,
                 int window=1, int update=0):
        
        self.name = 'GMM'
        self.n_components = n_components
        self.reg_covar = reg_covar
        self.n_jobs = n_jobs
        self.kappa = kappa
        self.window = window
        self.update = update
        
        self._is_initialized = 0
        self.iteration = 0
        
        BaseMixture._check_common_parameters(self)

        
    cdef void _initialize_temporary_arrays(self,double [:,:] points):
        '''
        Initialize all temporary arrays that will be used during the other methods
        '''
        dim = points.shape[1]
        
        # Initialize temporary memoryviews
        self.N_temp = np.zeros((1,self.n_components))
        self.N_temp2 = np.zeros((1,self.n_components))
        self.X_temp_fortran = np.zeros((self.n_components,dim))
        self.X_temp = np.zeros((self.n_components,dim))
        self.S_temp = np.zeros((self.n_components,dim,dim))
        self.cov_temp = np.zeros((dim,dim))
        self.points_temp2 = np.zeros((self.window,dim))
        self.points_temp = np.zeros((self.window,dim))
        self.mean_temp = np.zeros((1,dim))
        self.log_prob_norm = np.zeros((self.window,1))
        self.resp_temp = np.zeros((self.window,self.n_components))
        
            
    @cython.initializedcheck(False)
    def initialize(self, points_py):
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
        cdef double [:,:] points = points_py.copy()
        cdef int n_points = points.shape[0]
        cdef int dim = points.shape[1]
        
        # Initialization of temporary arrays
        self._initialize_temporary_arrays(points)
        cdef double [:,:] resp = np.zeros((n_points,self.n_components))
        cdef double [:,:] points_temp = np.zeros((n_points,dim))
        cdef double [:,:] points_temp2 = np.zeros((n_points,dim))

        # Parameters
        self.means = initialization_plus_plus(self.n_components,points)
        BaseMixture._initialize_cov(self,points,resp,points_temp,points_temp2)
        self.cov_chol = cvarray(shape=(self.n_components,dim,dim),itemsize=sizeof(double),format='d')
        BaseMixture._compute_cholesky_matrices(self)
        BaseMixture._initialize_weights(self,points,resp,points_temp,points_temp2)
        
        # Sufficient statistics
        self.N = cvarray(shape=(1,self.n_components),itemsize=sizeof(double),format='d')
        self.X = cvarray(shape=(self.n_components,dim),itemsize=sizeof(double),format='d')
        self.S = cvarray(shape=(self.n_components,dim,dim),itemsize=sizeof(double),format='d')        
        exp2D(self.log_weights,1,self.n_components,self.N)
        multiply2Dbyvect2D(self.means,self.n_components,dim,self.N,0,self.X)
        multiply3Dbyvect2D(self.cov,self.n_components,dim,dim,self.N,self.S)
        
        self.iteration = n_points + 1
        self._is_initialized = 1

        
    @cython.initializedcheck(False)
    cdef void _step_E_gen(self, double [:,:] points, double [:,:] log_resp,
                double [:,:] points_temp_fortran, double [:,:] points_temp,
                double [:,:] log_prob_norm):
        
        n_points = points.shape[0]
        """
        This method can be used whatever the number of points (for example in
        predict_log_resp)
        
        Parameters
        ----------
        points : an array (n_points,dim)
        log_resp : an array (n_points,n_components)
            an array containing the logarithm of the responsibilities.
        points_temp_fortran : an array (n_points,dim)
            a temporary array
        points_temp : an array (n_points,dim)
            a temporary array
        log_prob_norm : an array (n_points,)
            logarithm of the probability of each sample in points
        
        """
        _log_normal_matrix(points,self.means,self.cov_chol,log_resp,
                           self.cov_temp,points_temp_fortran,
                           points_temp,self.mean_temp)
        
        add2Dand2D(log_resp,n_points,self.n_components,
                   self.log_weights,1,self.n_components,log_resp)
        
        logsumexp_axis(log_resp,n_points,self.n_components,1,log_prob_norm)
        soustract2Dby2D(log_resp,n_points,self.n_components,
                        log_prob_norm,n_points,1,
                        log_resp)
            
    @cython.initializedcheck(False)
    cdef void _step_E(self,double [:,:] points,double [:,:] log_resp):
        '''
        This method may only be used in the fit method as the temporary arrays
        have the size of the model (self.window,dim)
        '''
        self._step_E_gen(points,log_resp,
                         self.points_temp,self.points_temp2,
                         self.log_prob_norm)

            
    @cython.initializedcheck(False)
    cdef void _step_M(self):
        """
        In this step the algorithm updates the values of the parameters
        (log_weights, means, covariances).
            
        """
        cdef int dim = self.means.shape[1]
        # We deduce the logarithm of weights from N which is the proportion of
        # points in each cluster.
        log2D(self.N,1,self.n_components,self.log_weights)
        
        divide2Dbyvect2D(self.X,self.n_components,dim,self.N,self.means)
        divide3Dbyvect2D(self.S,self.n_components,dim,dim,self.N,self.cov)
        
        if self.update:
            sqrt2D(self.N,1,self.n_components,self.N_temp)
            divide3Dbyvect2D(self.cov_chol,self.n_components,dim,dim,self.N_temp,self.cov_chol)
        else:
            self._compute_cholesky_matrices()
        
            
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef void _sufficient_statistics(self,double [:,:] points,double [:,:] log_resp):
        """
        This method is updating the sufficient statistics N,X and S.
        
        Parameters
        ----------
        points : an array (window,dim)
            an array containing the points
        log_resp : an array (window,n_components)
            an array containing the logarithm of the responsibilities.

        """
        cdef int dim = points.shape[1]

        exp2D(log_resp,self.window,self.n_components,self.resp_temp)
        
        cdef double gamma = 1/(self.iteration**self.kappa)
        cdef double window_double = self.window
        cdef double scalar_update
        cdef int i
        
        # New sufficient statistics
        add2Dscalar_reduce(self.resp_temp,self.window,self.n_components,1e-15,self.N_temp)
        divide2Dbyscalar(self.N_temp,1,self.n_components,window_double,self.N_temp)

        dot_spe_c(self.resp_temp,self.window,self.n_components,points,self.window,dim,self.X_temp_fortran)
        transpose_spe_f2c(self.X_temp_fortran,self.n_components,dim,self.X_temp)
        divide2Dbyscalar(self.X_temp,self.n_components,dim,window_double,self.X_temp)
        
        sqrt2D(self.resp_temp,self.window,self.n_components,self.resp_temp)
        for i in xrange(self.n_components):
            # diff is in points_temp
            # diff_weighted is in points_temp2
            soustract2Dby2D_idx(points,self.window,dim,self.means,0,i,self.points_temp)
            multiply2Dby2D_idx(self.points_temp,self.window,dim,
                               self.resp_temp,1,i,self.points_temp2)
            dot_spe_c(self.points_temp2,self.window,dim,
                      self.points_temp2,self.window,dim,self.cov_temp)
            transpose_spe_f2c_and_write(self.cov_temp,dim,dim,self.S_temp,i)

            if self.update:
                # points_temp2 (diff_weighted) is recquired in order to update cov_chol, so we begin
                # its update here
                scalar_update = sqrt(gamma/((1-gamma)*self.N[0,i]*self.window))
                multiply2Dbyscalar(self.points_temp2,self.window,dim,
                                   scalar_update,self.points_temp2)
                cholupdate(self.cov_chol,i,self.points_temp2,self.points_temp)


        divide3Dbyscalar(self.S_temp,self.n_components,dim,dim,
                         window_double,self.S_temp)
        
        if self.update:
            scalar_update = sqrt(1-gamma)
            sqrt2D(self.N,1,self.n_components,self.N_temp2)
            multiply2Dbyscalar(self.N_temp2,1,self.n_components,
                               scalar_update,self.N_temp2)
            multiply3Dbyvect2D(self.cov_chol,self.n_components,dim,dim,
                               self.N_temp2,self.cov_chol)

        # Sufficient statistics update
        update2D(self.N_temp,1,self.n_components,gamma,self.N)
        update2D(self.X_temp,self.n_components,dim,gamma,self.X)
        update3D(self.S_temp,self.n_components,dim,dim,gamma,self.S)
        
        
    cdef double _convergence_criterion(self,double [:,:] points,
                                       double [:,:] log_resp,
                                       double [:,:] log_prob_norm):
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
        cdef int n_points = log_prob_norm.shape[0]
        return sum2D(log_prob_norm,n_points,1)
#    
#    def _get_parameters(self):
#        return (self.N, self.X, self.S)
#    
#
#    def _set_parameters(self, params,verbose=True):
#        self.N, self.X, self.S = params
#        
#        real_components = len(self.X)
#        if self.n_components != real_components and verbose:
#            print('The number of components changed')
#        self.n_components = real_components
        
            
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
#
#        idx_existing = np.where(exist==1)
#        
#        log_weights = self.log_weights[idx_existing]
#        means = self.means[idx_existing]
#        cov = self.cov[idx_existing]
#                
#        params = (log_weights, means, cov)
#        
#        return params    