# encoding: utf-8
# cython: profile=True
# filename: kmeans_cython.pyx

import numpy as np
cimport numpy as np
from megamix.batch.initializations import initialization_plus_plus
from megamix.online.base import _check_saving
import h5py
import time
import warnings

import cython
from cython.view cimport array as cvarray
from basic_operations cimport update1D, update2D
from basic_operations cimport divide2Dbyscalar, divide2Dbyvect2D
from basic_operations cimport multiply2Dbyvect2D, subtract2Dby2D
from basic_operations cimport initialize, argmin, add2Dscalar_reduce, dot_spe_c, transpose_spe_f2c
from basic_operations cimport norm_axis1, norm_axis1_matrix, true_slice, log2D

from scipy.linalg.cython_blas cimport dgemm

def dist_matrix(double [:,:] points, double [:,:] means):
    """
    Wrapper for python
    """
    cdef int n_points = points.shape[0]
    cdef int n_components = means.shape[0]
    cdef int dim = means.shape[1]
    cdef double [:,:] dist = cvarray(shape=(n_components,dim),itemsize=sizeof(double),format='d')
    cdef double [:,:] dist_matrix = cvarray(shape=(n_points,n_components),itemsize=sizeof(double),format='d')
    
    dist_matrix_update(points,means,dist,dist_matrix)
    
    return np.asarray(dist_matrix)

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
        subtract2Dby2D(points,window,dim,means,n_components,dim,dist)
        norm_axis1(dist,n_components,dim,dist_matrix)

cdef class Kmeans:
                
    def __init__(self,int n_components=1,double kappa=1.0,
                 int window=1):
        
        self.name = 'Kmeans'
        self.init = 'usual'
        self.n_components = n_components
        self.kappa = kappa
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
    @cython.cdivision(True)
    def initialize(self,double [:,:] points):
        cdef int n_points = points.shape[0]
        cdef int dim = points.shape[1]
        
        if self.init=='usual':
            self.means = initialization_plus_plus(self.n_components,points)
            self.iteration = n_points + 1
            self.log_weights = np.zeros((1,self.n_components)) - np.log(self.n_components)
            #TODO real weights

        self.N = np.exp(self.log_weights)
        self.X = np.empty((self.n_components,dim),dtype=float)
        multiply2Dbyvect2D(self.means,self.n_components,dim,self.N,0,self.X)
        
        # initialize temporary memoryviews here
        self.N_temp = cvarray(shape=(1,self.n_components),itemsize=sizeof(double),format='d')
        self.X_temp = cvarray(shape=(self.n_components,dim),itemsize=sizeof(double),format='d')
        self.X_temp_fortran = cvarray(shape=(self.n_components,dim),itemsize=sizeof(double),format='d')
        self.dist_matrix = cvarray(shape=(self.window,self.n_components),itemsize=sizeof(double),format='d')
        self.dist = cvarray(shape=(self.n_components,dim),itemsize=sizeof(double),format='d')
        
        self.convergence_criterion_test = []
        self._is_initialized = 1

       
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function  
    @cython.initializedcheck(False)
    cdef void _step_E_gen(self, double [:,:] points, double [:,:] assignements,
                      double [:,:] dist_matrix):
        cdef int n_points = points.shape[0]
        cdef int dim = points.shape[1]
        initialize(assignements,n_points,self.n_components)
        
        dist_matrix_update(points,self.means,self.dist,dist_matrix)

        cdef int index_min,i
        for i in xrange(n_points):
            index_min = argmin(dist_matrix,i,self.n_components)
            assignements[i,index_min] = 1
        
    
    def _step_E(self,points):
        """
        Wrapper for python tests
        """
        cdef int n_points = points.shape[0]
        cdef double [:,:] assignements = cvarray(shape=(n_points,self.n_components),itemsize=sizeof(double),format='d')
        cdef double [:,:] dist_matrix = cvarray(shape=(n_points,self.n_components),itemsize=sizeof(double),format='d')
        
        self._step_E_gen(points,assignements,dist_matrix)
        
        return np.asarray(assignements)
                    
    
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function  
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cpdef void _step_M(self, double[:,:] points, double[:,:] assignements):
        cdef int dim = points.shape[1]
        cdef double window_double = self.window

        # New sufficient statistics
        add2Dscalar_reduce(assignements,self.window,self.n_components,1e-15,self.N_temp)
        divide2Dbyscalar(self.N_temp,1,self.n_components,window_double,self.N_temp)
        dot_spe_c(assignements,self.window,self.n_components,points,self.window,dim,self.X_temp_fortran)
        transpose_spe_f2c(self.X_temp_fortran,self.n_components,dim,self.X_temp)
        divide2Dbyscalar(self.X_temp,self.n_components,dim,window_double,self.X_temp)
        
        # Sufficient statistics update
        cdef double gamma
        gamma = 1/((self.iteration + self.window//2)**self.kappa)
        
        update2D(self.N_temp,1,self.n_components,gamma,self.N)
        update2D(self.X_temp,self.n_components,dim,gamma,self.X)

        # Parameter update
        divide2Dbyvect2D(self.X,self.n_components,dim,self.N,self.means)
        log2D(self.N,1,self.n_components,self.log_weights)
        
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    @cython.initializedcheck(False)
    def score(self,points,assignements=None):
        cdef int n_points = points.shape[0]
        cdef double distortion = 0
        cdef int i
        
        if assignements is None:
            assignements = self.predict_assignements(points)
            
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
        
        
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function        
    @cython.initializedcheck(False)
    def fit(self,double [:,:] points_data, points_test=None,
            saving=None, str file_name='model', check_convergence_iter=None,
            int saving_iter=2):
        
        cdef int n_points = points_data.shape[0]
        cdef int dim = points_data.shape[1]
        cdef double [:,:] resp = np.zeros((self.window,self.n_components))
        cdef double [:,:] point = np.zeros((self.window,dim))
        cdef int i
        
        # Early stopping preparation
        test_exists = points_test is not None
        if test_exists:
            if check_convergence_iter is None:
                raise ValueError('A value must be given for check_convergence_iter')
            elif not isinstance(check_convergence_iter,int) or check_convergence_iter < 1:
                raise ValueError('check_convergence_iter must be a positive int')
        self.convergence_criterion_test.append(self.score(points_test))
        
        condition = _check_saving(saving,saving_iter)
        
        if self._is_initialized:
            for i in xrange(n_points//self.window):
                true_slice(points_data,i,dim,point,self.window)
                self._step_E_gen(point,resp,self.dist_matrix)
                self._step_M(point,resp)
                self.iteration += 1
                
                # Checking early stopping
                if test_exists and (i+1)%check_convergence_iter == 0:
                    self.convergence_criterion_test.append(self.score(points_test))
                    change = self.convergence_criterion_test[-2] - self.convergence_criterion_test[-1]
                    if change < 0:
                        best_params = self._get_parameters()
                    else:
                        print('Convergence was reached at iteration', self.iter)
                        self._set_parameters(best_params)
                        break
                
                if condition(i+1):
                    f = h5py.File(file_name + '.h5', 'a')
                    grp = f.create_group('iter' + str(self.iteration))
                    self.write(grp)
                    f.close()
        else:
            raise ValueError('The model must be initialized')

    
    def get(self,name):
        if name=='_is_initialized':
            return self._is_initialized
        if name=='log_weights':
            return np.array(self.log_weights).reshape(self.n_components)
        if name=='means':
            return np.array(self.means)
        if name=='N':
            return np.array(self.N).reshape(self.n_components)
        if name=='X':
            return np.array(self.X)
        if name=='iter':
            return self.iteration
        elif name=='window':
            return self.window
        elif name=='kappa':
            return self.kappa
        elif name=='name':
            return self.name
                
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
            self._step_E_gen(points,resp,dist_matrix)
            return resp

        else:
            raise Exception("The model is not initialized")

    def write(self,group):
        """
        A method creating datasets in a group of an hdf5 file in order to save
        the model
        
        Parameters
        ----------
        group : HDF5 group
            A group of a hdf5 file in reading mode

        """
        dim = self.means.shape[1]
        group.create_dataset('means',(self.n_components,dim),dtype='float64')
        group['means'][...] = np.array(self.means)
        group.create_dataset('log_weights',(1,self.n_components),dtype='float64')
        group['log_weights'][...] = np.array(self.log_weights)
        group.attrs['iter'] = self.iteration
        group.attrs['time'] = time.time()
        
    def read_and_init(self,group,points):
        """
        A method reading a group of an hdf5 file to initialize DPGMM
        
        Parameters
        ----------
        group : HDF5 group
            A group of a hdf5 file in reading mode
            
        """
        self.means = np.asarray(group['means'].value)
        self.log_weights = np.asarray(group['log_weights'].value).reshape(1,self.n_components)
        self.iteration = group.attrs['iter']

        n_components = len(self.means)
        if n_components != self.n_components:
            warnings.warn('You are now currently working with %s components.'
                          % n_components)
            self.n_components = n_components
        
        self.init = 'user'
        
        self.initialize(points)