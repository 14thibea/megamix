# encoding: utf-8
# cython: profile=True
# filename: kmeans_cython.pyx

import numpy as np
cimport numpy as np
from megamix.batch.initializations import initialization_plus_plus

import cython
from cython.view cimport array as cvarray
from basic_operations cimport update1D, update2D
from basic_operations cimport divide2Dbyscalar, divide2Dbyvect2D
from basic_operations cimport multiply2Dbyvect2D, soustract2Dby2D
from basic_operations cimport initialize, argmin, add2Dscalar_reduce, dot_spe_c, transpose_spe_f2c
from basic_operations cimport norm_axis1, norm_axis1_matrix, true_slice

from scipy.linalg.cython_blas cimport dgemm

 
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function  
@cython.initializedcheck(False)
cdef void dist_matrix_update(double [:,:] points, double [:,:] means,
                             double [:,:] dist, double [:,:] dist_matrix):
    '''
    Parameters
    ----------
    points : array (n_points,dim)
    means : array (n_components,dim)
    dist : array (n_components,dim)
    dist_matrix : array (n_points,n_components)
    
    '''
    cdef int window = points.shape[0]
    cdef int dim = points.shape[1]
    cdef int n_components = means.shape[0]
    
    if window > 1:
        norm_axis1_matrix(points,window,dim,
                          means,n_components,dim,
                          dist,dist_matrix)
    else:
        soustract2Dby2D(points,window,dim,means,n_components,dim,dist)
        norm_axis1(dist,n_components,dim,dist_matrix)

cdef class Kmeans:
                
    def __init__(self,int n_components=1,int n_jobs=1,double kappa=1.0,
                 int window=1):
        
        self.name = 'Kmeans'
        self.n_components = n_components
        self.kappa = kappa
        self.n_jobs = n_jobs
        self.window = window
        
        self._is_initialized = 0
        self.iteration = 0
        
        self._check_parameters()


    cdef void _check_parameters(self):
        
        if self.n_components < 1:
            raise ValueError("The number of components cannot be less than 1")
        else:
            self.n_components = int(self.n_components)
            
        if self.kappa <= 0 or self.kappa > 1:
            raise ValueError("kappa must be in ]0,1]")

       
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function  
    @cython.initializedcheck(False)
    def _step_E(self, double [:,:] points, int dim,
                      double [:,:] assignements, double [:,:] dist_matrix):
        cdef int n_points = points.shape[0]
        initialize(assignements,n_points,self.n_components)
        
        dist_matrix_update(points,self.means,self.dist,dist_matrix)

        cdef int index_min,i
        for i in xrange(n_points):
            index_min = argmin(dist_matrix,i,self.n_components)
            assignements[i,index_min] = 1
        
        
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function  
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    def _step_M(self, double[:,:] points, int dim, double[:,:] assignements):
        cdef double window_double = self.window

        # New sufficient statistics
        add2Dscalar_reduce(assignements,self.window,self.n_components,1e-15,self.N_temp)
        divide2Dbyscalar(self.N_temp,1,self.n_components,window_double,self.N_temp)
        dot_spe_c(assignements,self.window,self.n_components,points,self.window,dim,self.X_temp_fortran)
        transpose_spe_f2c(self.X_temp_fortran,self.n_components,dim,self.X_temp)
        divide2Dbyscalar(self.X_temp,self.n_components,dim,window_double,self.X_temp)
        
        # Sufficient statistics update
        cdef double gamma
        gamma = 1/(self.iteration**self.kappa)
        
        update2D(self.N_temp,1,self.n_components,gamma,self.N)
        update2D(self.X_temp,self.n_components,dim,gamma,self.X)

        # Parameter update
        divide2Dbyvect2D(self.X,self.n_components,dim,self.N,self.means)
        
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    @cython.initializedcheck(False)
    def distortion(self,points,assignements):
        from megamix.kmeans import dist_matrix
        cdef int n_points = points.shape[0]
        cdef double distortion = 0
        cdef int i
        
        if self._is_initialized:
            means = np.asarray(self.means)
            for i in xrange(self.n_components):
                assignements_i = assignements[:,i:i+1]
                n_set = np.sum(assignements_i)
                idx_set,_ = np.where(assignements_i==1)
                sets = points[idx_set]
                if n_set != 0:
                    M = np.linalg.norm(sets-means[i],axis=1)
                    distortion += np.sum(M)
                
            return distortion

        else:
            raise Exception("The model is not initialized")
            
            
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function  
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    def initialize(self,double [:,:] points):
        cdef int n_points = points.shape[0]
        cdef int dim = points.shape[1]
        
        self.means = initialization_plus_plus(self.n_components,points)
        self.N = 1./self.n_components * np.ones((1,self.n_components)) #TODO real weights
        self.X = np.empty((self.n_components,dim),dtype=float)
        multiply2Dbyvect2D(self.means,self.n_components,dim,self.N,0,self.X)
        
        # initialize temporary memoryviews here
        self.N_temp = cvarray(shape=(1,self.n_components),itemsize=sizeof(double),format='d')
        self.X_temp = cvarray(shape=(self.n_components,dim),itemsize=sizeof(double),format='d')
        self.X_temp_fortran = cvarray(shape=(self.n_components,dim),itemsize=sizeof(double),format='d')
        self.dist_matrix = cvarray(shape=(self.window,self.n_components),itemsize=sizeof(double),format='d')
        self.dist = cvarray(shape=(self.n_components,dim),itemsize=sizeof(double),format='d')
        
        self.iteration = n_points + 1
        self._is_initialized = 1
        
        
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function        
    @cython.initializedcheck(False)
    def fit(self,double [:,:] points):
        cdef int n_points = points.shape[0]
        cdef int dim = points.shape[1]
        cdef double [:,:] resp = np.zeros((self.window,self.n_components))
        cdef double [:,:] point = np.zeros((self.window,dim))
        
        cdef int i
        if self._is_initialized:
            for i in xrange(n_points//self.window):
                true_slice(points,i,dim,point,self.window)
                self._step_E(point,dim,resp,self.dist_matrix)
                self._step_M(point,dim,resp)
                self.iteration += 1
        else:
            raise ValueError('The model must be initialized')

    
    def get(self,name):
        if name=='means':
            return np.array(self.means)
        if name=='N':
            return np.array(self.N)
        if name=='N_temp':
            return np.array(self.N_temp)
        if name=='X':
            return np.array(self.X)
        if name=='iter':
            return self.iteration
        elif name=='window':
            return self.window
                
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function        
    def predict_assignements(self,points):
        '''
        Parameters
        ----------
        points : array (n_points,dim)
            An array of points
        
        Returns
        -------
        assignements : array (n_points,n_components)
            The hard assignements of each point to one cluster
        '''
        cdef int n_points = points.shape[0]
        cdef int dim = points.shape[1]
        resp = np.zeros((n_points,self.n_components))
        cdef double [:,:] dist_matrix = np.zeros((n_points,self.n_components))

        if self._is_initialized:
            self._step_E(points,dim,resp,dist_matrix)
            return resp

        else:
            raise Exception("The model is not initialized")
