# -*- coding: utf-8 -*-
#
#Created on Fri Apr 21 11:13:09 2017
#
#author: Elina THIBEAU-SUTRE
#
#cython: profile=True
from megamix.online.base import _check_saving
import numpy as np
import scipy.linalg
from scipy.special import gammaln
import os
import warnings
import time
import h5py
from kmeans_cython cimport dist_matrix_update

from scipy.linalg.cython_blas cimport drot
from scipy.linalg.cython_lapack cimport dpotrf
import cython
from cython.view cimport array as cvarray
from libc.math cimport log,pi,exp,hypot
from basic_operations cimport true_slice, writecol_sum_square, initialize,transpose_spe_f2c
from basic_operations cimport dot_spe_c2, log_det_tr, triangular_inverse_cov
from basic_operations cimport add2Dscalar_reduce, add2Dscalar, add2Dscalar_col_i
from basic_operations cimport subtract2Dby2D_idx, subtract2Dby2D
from basic_operations cimport multiply2Dbyscalar, multiply2Dby2D_idx, multiply3Dbyscalar
from basic_operations cimport divide2Dbyscalar, divide3Dbyvect2D
from basic_operations cimport reg_covar, transpose_spe_f2c_and_write, dot_spe_c,argmin
from basic_operations cimport cast3Din2D,cast2Din3D,erase_above_diag

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef void cholupdate(double [:,:,:] cov_chol, int idx,
               double [:,:] points, double [:,:] points_temp):
    '''
    Update the lower triangular Cholesky factor cov_chol with the rank 1 addition
    implied by x such that:
    <cov_chol_new.T,cov_chol_new> = <cov_chol.T,cov_chol> + sum_i(outer(points[i],points[i]))
    '''
    cdef unsigned int n_points,dim
    cdef unsigned int i,j,k
    cdef double r,c,s,a,b
    cdef double c_,s_

    n_points = points.shape[0]
    dim = points.shape[1]
    
    points_temp = points.copy()
    
    for j in xrange(n_points):
        for k in xrange(dim):
            r = hypot(cov_chol[idx,k,k], points_temp[j,k]) # drotg
            c = r / cov_chol[idx,k,k]
            s = points_temp[j,k] / cov_chol[idx,k,k]
            cov_chol[idx,k,k] = r
            #TODO: Use BLAS drot instead of inner for loop
            for i in xrange((k+1),dim):
                cov_chol[idx,i,k] = (cov_chol[idx,i,k] + s * points_temp[j,i]) / c
                points_temp[j,i] = c * points_temp[j,i] - s * cov_chol[idx,i,k]

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void logsumexp_axis(double[:,:] a, int a_x, int a_y, int axis, double [:,:] result):
    cdef int x,y
    cdef double max_exp
    cdef double sum_
    
    if axis==1:
        for x in xrange(a_x):
            max_exp = a[x,0]
            sum_ = 0.0
            for y in xrange(a_y):
                if a[x,y] > max_exp:
                    max_exp = a[x,y]
                    
            for y in xrange(a_y):
                sum_ += exp(a[x,y] - max_exp)
            
            result[x,0] = log(sum_) + max_exp
                  
    elif axis==0:
        for y in xrange(a_y):
            max_exp = a[0,y]
            sum_ = 0.0
            for x in xrange(a_x):
                if a[x,y] > max_exp:
                    max_exp = a[x,y]
                    
            for x in xrange(a_x):
                sum_ += exp(a[x,y] - max_exp)
            
            result[0,y] = log(sum_) + max_exp


cdef void _log_normal_matrix(double [:,:] points,double [:,:] means,
                             double [:,:,:] cov_chol,double [:,:] log_normal_matrix,
                             double [:,:] cov_temp,double [:,:] points_temp_fortran,
                             double [:,:] points_temp,double [:,:] mean_temp):
    """
    This method computes the log of the density of probability of a normal law centered. Each line
    corresponds to a point from points.
    
    :param points: an array of points (n_points,dim)
    :param means: an array of k points which are the means of the clusters (n_components,dim)
    :param cov: an array of k arrays which are the covariance matrices (n_components,dim,dim)
    :return: an array containing the log of density of probability of a normal law centered (n_points,n_components)
    """
    cdef int n_points = points.shape[0]
    cdef int dim = points.shape[1]
    cdef int n_components = means.shape[0]
    cdef double coeff = dim*log(2*pi)
    cdef int i
    cdef double log_det_prec
    
    initialize(log_normal_matrix,n_points,n_components) # TODO useless ?
    for i in xrange(n_components):
        # Writing the cholesky factor of the precision matrix in cov_temp
        triangular_inverse_cov(cov_chol,i,dim,cov_temp)
        # Computing of the determinant of the precision matrix
        log_det_prec = log_det_tr(cov_temp,dim)
        # Writing the dot product between the points and the cholesky factor
        # of the precision matrix i in points_temp (n_points,dim)
        dot_spe_c2(points,n_points,dim,cov_temp,dim,dim,points_temp_fortran)
        transpose_spe_f2c(points_temp_fortran,n_points,dim,points_temp)
        # Writing the dot product between the mean i and the cholesky factor
        # of the precision matrix i in mean_temp (1,dim)
#        true_slice(means,i,dim,mean_temp,1)
        dot_spe_c2(means[i:i+1],1,dim,cov_temp,dim,dim,mean_temp)
        # Writing the difference between them in points_temp (n_points,dim)
        subtract2Dby2D(points_temp,n_points,dim,mean_temp,1,dim,points_temp)
        # Writing the contents of the exponential in the ith column of log_normal_matrix
        writecol_sum_square(points_temp,n_points,dim,1,i,log_normal_matrix)
        # Add log_det_prec to the ith column
        add2Dscalar_col_i(log_normal_matrix,n_points,i,-2*log_det_prec,log_normal_matrix)
        
    #Add coeff and multiply by 0.5            
    add2Dscalar(log_normal_matrix,n_points,n_components,coeff,log_normal_matrix)
    divide2Dbyscalar(log_normal_matrix,n_points,n_components,-2.,log_normal_matrix)

def _log_B(W,nu):
    """
    The log of a coefficient involved in the Wishart distribution
    see Bishop book p.693 (B.78)
    """
    
    dim,_ = W.shape
    
    det_W = np.linalg.det(W)
    log_gamma_sum = np.sum(gammaln(.5 * (nu - np.arange(dim)[:, np.newaxis])), 0)
    result = - nu*0.5*np.log(det_W) - nu*dim*0.5*np.log(2)
    result += -dim*(dim-1)*0.25*np.log(np.pi) - log_gamma_sum
    return result


def _log_C(alpha):
    """
    The log of a coefficient involved in the Dirichlet distribution
    see Bishop book p.687 (B.23)
    """
    
    return gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))

cdef class BaseMixture:
#    """
#    Base class for mixture models.
#    This abstract class specifies an interface for other mixture classes and
#    provides basic common methods for mixture models.
#    """
    
    # Abstract methods
    cdef void _step_E_gen(self, double [:,:] points, double [:,:] log_resp,
                double [:,:] points_temp_fortran, double [:,:] points_temp,
                double [:,:] log_prob_norm):
        pass
    
    cdef void _cstep_E(self,double [:,:] points,double [:,:] log_resp):
        pass

    cpdef void _step_M(self):
        pass
    
    cpdef void _sufficient_statistics(self,double [:,:] points,double [:,:] log_resp):
        pass
    
    cdef double _convergence_criterion(self,double [:,:] points,
                           double [:,:] log_resp,
                           double [:,:] log_prob_norm):
        pass
    
    
    cdef void _check_common_parameters(self):
        """
        This function tests the parameters common to all algorithms
        
        """
        
        if self.n_components < 1:
            raise ValueError("The number of components cannot be less than 1")
            
            
    cdef void _check_prior_parameters(self, points):
        """
        This function tests the hyperparameters of the VBGMM and the DBGMM
        The parameters points must be a python array and not a memoryview !
        
        """
        cdef int n_points = points.shape[0]
        cdef int dim = points.shape[1]
        print('I am here')
        print(self.alpha_0_flag)
        
        #Checking alpha_0
        if self.alpha_0_flag == 1:
            print('I am there')
            self.alpha_0 = 1./self.n_components
        elif self.alpha_0 < 0:
            raise ValueError("alpha_0 must be positive")
        
        #Checking beta_0
        if self.beta_0_flag == 1:
            self.beta_0 = 1.0
        
        #Checking nu_0
        if self.nu_0_flag == 1:
            self.nu_0 = dim
        
        elif self.nu_0 < dim:
            raise ValueError("nu_0 must be more than the dimension of the"
                             "problem or the gamma function won't be defined")
        
        
        #Checking prior mean
        if self.means_prior_flag == 1:
            self.means_prior = np.mean(points,axis=0)
        elif self._means_prior.shape[1] != dim:
            raise ValueError("the mean prior must have the same dimension as "
                             "the points : %s."
                             % dim)
        
        # Checking prior W-1
        if self.inv_prec_prior_flag == 1:
            self.inv_prec_prior = np.cov(points.T)
        elif self.inv_prec_prior.shape != (dim,dim):
            raise ValueError("the covariance prior must have the same "
                             "dimension as the points : %s."
                             % dim)
            
        
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
    cdef void _cinitialize_cov(self,double [:,:] points, double [:,:] assignements,
                              double [:,:] diff, double [:,:] diff_weighted):
        '''
        Initialize the attribute cov
        Must be initialized :
            * all temporary arrays
            * means
        
        Parameters
        ----------
        points : an array (n_points,dim)
        resp : an array (n_points,n_components)
        diff : an array (n_points,dim)
        diff_weighted : an array (n_points,dim)
        
        '''
        
        cdef int n_points = points.shape[0]
        cdef int dim = points.shape[1]
        cdef double[:,:] dist_matrix = cvarray(shape=(n_points,self.n_components),itemsize=sizeof(double),format='d')
        cdef int i, index_min
        cdef double scalar = 1./n_points

        
        # Computation of the matrix of distances
        dist_matrix_update(points,self.means,self.X_temp,dist_matrix)
        
        # Computation of hard assignements
        for i in xrange(n_points):
            index_min = argmin(dist_matrix,i,self.n_components) #the cluster number of the ith point is index_min
            assignements[i,index_min] = 1
        
        # Computation of N
        add2Dscalar_reduce(assignements,n_points,self.n_components,1e-15,self.N_temp)
        multiply2Dbyscalar(self.N_temp,1,self.n_components,scalar,self.N_temp)

        # Computation of S 
        self.cov = cvarray(shape=(self.n_components,dim,dim),itemsize=sizeof(double),format='d')        
        for i in xrange(self.n_components):
            subtract2Dby2D_idx(points,n_points,dim,self.means,0,i,diff)
            multiply2Dby2D_idx(diff,n_points,dim,assignements,1,i,diff_weighted)
            dot_spe_c(diff_weighted,n_points,dim,diff,n_points,dim,self.cov_temp)
            transpose_spe_f2c_and_write(self.cov_temp,dim,dim,self.S_temp,i)
            reg_covar(self.S_temp,i,dim,self.reg_covar)
        
        multiply3Dbyscalar(self.S_temp,self.n_components,dim,dim,scalar,self.S_temp)
        # We now have the equivalent of S in self.S_temp
        
        divide3Dbyvect2D(self.S_temp,self.n_components,dim,dim,self.N_temp,self.cov)
        
    
    def _initialize_cov(self,points):
        self._initialize_temporary_arrays(points)
        cdef int n_points = points.shape[0]
        cdef int dim = points.shape[1]
        cdef double [:,:] assignements = cvarray(shape=(n_points,self.n_components),itemsize=sizeof(double),format='d')
        cdef double [:,:] points_temp = cvarray(shape=(n_points,dim),itemsize=sizeof(double),format='d')
        cdef double [:,:] points_temp2 = cvarray(shape=(n_points,dim),itemsize=sizeof(double),format='d')
        self._cinitialize_cov(points,assignements,points_temp,points_temp2)
        
    
    @cython.initializedcheck(False)
    cdef void _cinitialize_weights(self,double [:,:] points,double [:,:] log_normal_matrix,
                            double [:,:] points_temp, double [:,:] points_temp_fortran):
        '''
        Initialize the attribute log_weights
        Must be initialized :
            * all temporary arrays
            * means
            * cov
            * cov_chol
        
        Parameters
        ----------
        points : an array (n_points,dim)
        log_normal_matrix : an array (n_points,n_components)
        
        '''
        cdef int n_points = points.shape[0]
        cdef int dim = points.shape[1]
        cdef double norm = -log(n_points)
        cdef double [:,:] log_prob_norm = np.zeros((n_points,1))
        
        _log_normal_matrix(points,self.means,self.cov_chol,log_normal_matrix,
                           self.cov_temp,points_temp_fortran,
                           points_temp,self.mean_temp)
        logsumexp_axis(log_normal_matrix,n_points,self.n_components,1,log_prob_norm)
        # Now log_normal_matrix will contain the log of responsibilities as it 
        # is normed
        subtract2Dby2D(log_normal_matrix,n_points,self.n_components,
                        log_prob_norm,n_points,1,
                        log_normal_matrix)
        
        self.log_weights = cvarray(shape=(1,self.n_components),itemsize=sizeof(double),format='d')
        logsumexp_axis(log_normal_matrix,n_points,self.n_components,0,self.log_weights)
        add2Dscalar(self.log_weights,1,self.n_components,norm,self.log_weights)

    
    def _initialize_weights(self,points):
        """
        Wrapper for python tests
        """
        self._initialize_temporary_arrays(points)
        cdef int n_points = points.shape[0]
        cdef int dim = points.shape[1]
        cdef double [:,:] log_normal_matrix = cvarray(shape=(n_points,self.n_components),itemsize=sizeof(double),format='d')
        cdef double [:,:] points_temp = cvarray(shape=(n_points,dim),itemsize=sizeof(double),format='d')
        cdef double [:,:] points_temp2 = cvarray(shape=(n_points,dim),itemsize=sizeof(double),format='d')
        self._cinitialize_weights(points,log_normal_matrix,points_temp,points_temp2)
    

    def _check_points(self,points):
        """
        This method checks that the points have the same dimension than the
        problem
        
        """
        
        
        if len(points.shape) == 1:
            points = points.reshape(1,len(points))
            
        elif len(points.shape) != 2:
            raise ValueError('Only 2D or 1D arrays are admitted')

        dim_points = points.shape[1]
        means = self.get('means')
        dim_means = means.shape[1]
        if dim_means != dim_points:
            raise ValueError('The points given must have the same '
                             'dimension as the problem : ' + str(dim_means))
        return points
    
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.initializedcheck(False)
    cdef void _compute_cholesky_matrices(self):
        """
        Computes the cholesky matrices of the covariances
        
        Must be initialized :
            * all temporary arrays
            * cov
            * cov_chol
        """
        
        
        cdef int dim = self.cov.shape[1]
        cdef int lda = dim
        cdef int info = 0
        cdef int i
        
        for i in xrange(self.n_components):
            cast3Din2D(self.cov,i,dim,self.cov_temp)
            dpotrf('U',&dim,&self.cov_temp[0,0],&lda,&info) # TODO is there something faster ?
            # TODO savoir s'il est utile d'effacer les valeurs au-dessus de la diagonale
            # ou si on peut utiliser un produit matriciel triangulaire
            erase_above_diag(self.cov_temp,dim)
            cast2Din3D(self.cov_temp,i,dim,self.cov_chol)
    
    
    def fit(self,double [:,:] points,saving=None,str file_name='model',
            int saving_iter=2):
        """The EM algorithm
        
        Parameters
        ----------
        points : array (n_points,dim)
            A 2D array of points on which the model will be trained
            
        Returns
        -------
        None
        
        """
        cdef int n_points = points.shape[0]
        cdef int dim = points.shape[1]
        cdef double [:,:] log_resp = np.zeros((self.window,self.n_components))
        cdef double [:,:] point = np.zeros((self.window,dim))
        
        condition = _check_saving(saving,saving_iter)

        cdef int i
        if self._is_initialized:
            for i in xrange(n_points//self.window):
                true_slice(points,i,dim,point,self.window)
                self._cstep_E(point,log_resp)
                self._sufficient_statistics(point,log_resp)
                self._step_M()
                self.iteration += self.window
                
                if condition(i+1):
                    f = h5py.File(file_name + '.h5', 'a')
                    grp = f.create_group('iter' + str(self.iteration))
                    self.write(grp)
                    f.close()

        
        else:
            raise ValueError('The system has to be initialized.')
    
    
    def predict_log_resp(self,points):
        """
        This function returns the logarithm of each point's responsibilities
        
        Parameters
        ----------
        points : array (n_points_bis,dim)
            a 1D or 2D array of points with the same dimension as the problem
            
        Returns
        -------
        log_resp : array (n_points_bis,n_components)
            the logarithm of the responsibilities
            
        """
        
        points = self._check_points(points)
        n_points = points.shape[0]
        
        if self._is_initialized:
            log_resp = np.zeros((n_points,self.n_components))
            points_temp = np.zeros_like(points)
            points_temp2 = np.zeros_like(points)
            log_prob_norm = np.zeros((n_points,1))
            self._step_E_gen(points, log_resp,
                             points_temp2, points_temp, log_prob_norm)
            return log_resp

        else:
            raise Exception("The model is not initialized")

    
    def score(self, points_py):
        """
        This function return the score of the function, which is the logarithm of
        the likelihood for GMM and the logarithm of the lower bound of the likelihood
        for VBGMM and DPGMM
        
        Parameters
        ----------
        points : array (n_points_bis,dim)
            a 1D or 2D array of points with the same dimension as the problem
            
        Returns
        -------
        score : float
            
        """
        points_check = self._check_points(points_py)
        n_points = points_check.shape[0]
        dim = points_check.shape[1]
        cdef double [:,:] points = points_check.copy()
        cdef double [:,:] log_resp = np.zeros((n_points,self.n_components))
        cdef double [:,:] points_temp = np.zeros((n_points,dim))
        cdef double [:,:] points_temp2 = np.zeros((n_points,dim))
        cdef double [:,:] log_prob_norm = np.zeros((n_points,1))

        
        if self._is_initialized:
            
            self._step_E_gen(points, log_resp,
                             points_temp2, points_temp, log_prob_norm)
            score = self._convergence_criterion(points,log_resp,log_prob_norm)
            return score
        
        else:
            raise Exception("The model is not fitted")
    
    def write(self,group):
        """
        A method creating datasets in a group of an hdf5 file in order to save
        the model
        
        Parameters
        ----------
        group : HDF5 group
            A group of a hdf5 file in reading mode

        """
        dim = self.get('means').shape[1]
        group.create_dataset('log_weights',(self.n_components,),dtype='float64')
        group['log_weights'][...] = self.get('log_weights')
        group.create_dataset('means',(self.n_components,dim),dtype='float64')
        group['means'][...] = self.get('means')
        group.create_dataset('cov',(self.n_components,dim,dim),dtype='float64')
        group['cov'][...] = self.get('cov')
        group.attrs['iter'] = self.get('iter')
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
        
        try:
            self.cov = np.asarray(group['cov'].value)
        except KeyError:
            warnings.warn('You are reading a model with no covariance.'
                          'They will be initialized.')
            self.init = 'read_kmeans'
        
        self.initialize(points)


    def get(self,name):
        if name=='_is_initialized':
            return self._is_initialized
        elif name=='iter':
            return self.iteration
        elif name=='init':
            return self.init
        elif name=='name':
            return self.name
        elif name=='kappa':
            return self.kappa
        elif name=='window':
            return self.window
        elif name=='means':
            return np.asarray(self.means)
        elif name=='cov':
            return np.asarray(self.cov)
        elif name=='cov_chol':
            return np.asarray(self.cov_chol)
        elif name=='log_weights':
            return np.asarray(self.log_weights).reshape(self.n_components)
        elif name=='N':
            return np.asarray(self.N).reshape(self.n_components)
        elif name=='X':
            return np.asarray(self.X)
        elif name=='S':
            return np.asarray(self.S)
        elif name=='n_components':
            return self.n_components
        elif name=='upgrade':
            return self.upgrade
        
        # VBGMM only
        elif name=='alpha_0':
            return self.alpha_0
        elif name=='beta_0':
            return self.beta_0
        elif name=='nu_0':
            return self.nu_0
        elif name=='alpha':
            return np.asarray(self.alpha)
        elif name=='beta':
            return np.asarray(self.beta)
        elif name=='nu':
            return np.asarray(self.nu)
        elif name=='inv_prec_prior':
            return np.asarray(self.inv_prec_prior)
        elif name=='mean_prior':
            return np.asarray(self.mean_prior)
       
    def set(self,name,data):
        if name=='_is_initialized':
            self._is_initialized = data
        elif name=='iter':
            self.iteration = data
        elif name=='window':
            self.window = data
        elif name=='means':
            self.means = data
        elif name=='cov':
            self.cov = data
        elif name=='cov_chol':
            self.cov_chol = data
        elif name=='log_weights':
            self.log_weights = data
        elif name=='N':
            self.N = data.reshape(1,len(data))
        elif name=='X':
            self.X = data
        elif name=='S':
            self.S = data
        elif name=='n_components':
            self.n_components = data
        
        # VBGMM only
        elif name=='alpha_0':
            self.alpha_0 = data
        elif name=='beta_0':
            self.beta_0 = data
        elif name=='nu_0':
            self.nu_0 = data
        elif name=='alpha':
            self.alpha = data
        elif name=='beta':
            self.beta = data
        elif name=='nu':
            self.nu = data
        elif name=='inv_prec_prior':
            self.inv_prec_prior = data
        elif name=='mean_prior':
            self.mean_prior = data