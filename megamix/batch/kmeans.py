# -*- coding: utf-8 -*-
#
#Created on Mon Apr 10 14:20:27 2017
#
#author: Elina Thibeau-Sutre
#

import numpy as np
import h5py
from .base import BaseMixture, _check_saving

def dist_matrix(points,means):
    
    XX = np.einsum('ij,ij->i', points, points)[:, np.newaxis]    # Size (n_points,1)
    squared_matrix = np.dot(points,means.T)                      # Size (n_points,n_components)
    YY = np.einsum('ij,ij->i', means, means)[np.newaxis, :]      # Size (1,n_components)
    
    squared_matrix *= -2
    squared_matrix += XX
    squared_matrix += YY
    np.maximum(squared_matrix, 0, out=squared_matrix)    
    
    return np.sqrt(squared_matrix, out=squared_matrix)

class Kmeans(BaseMixture):

    """
    Kmeans model.
    
    Parameters
    ----------
    
    n_components : int, defaults to 1.
        Number of clusters used.
    
    init : str, defaults to 'kmeans'.
        Method used in order to perform the initialization,
        must be in ['random', 'plus', 'AF_KMC'].

    Attributes
    ----------
    
    name : str
        The name of the method : 'Kmeans'
        
    means : array of floats (n_components,dim)
        Contains the computed means of the model.
    
    log_weights : array of floats (n_components,)
        Contains the logarithm of the mixing coefficient of each cluster.
    
    iter : int
        The number of iterations computed with the method fit()
    
    _is_initialized : bool
        Ensures that the model has been initialized before using other
        methods such as distortion() or predict_assignements().
    
    Raises
    ------
    ValueError : if the parameters are inconsistent, for example if the cluster number is negative, init_type is not in ['resp','mcw']...
    
    References
    ----------
    'Fast and Provably Good Seedings for k-Means', O. Bachem, M. Lucic, S. Hassani, A.Krause
    'Lloyd's algorithm <https://en.wikipedia.org/wiki/Lloyd's_algorithm>'_
    'The remarkable k-means++ <https://normaldeviate.wordpress.com/2012/09/30/the-remarkable-k-means/>'_
 
    """
    def __init__(self,n_components=1,init="plus",n_jobs=1):
        
        super(Kmeans, self).__init__()

        self.name = 'Kmeans'
        self.n_components = n_components
        self.init = init
        self.n_jobs = n_jobs
        
        self._is_initialized = False
        self.iter = 0
        
        self._check_parameters()

    def _check_parameters(self):
        
        if self.n_components < 1:
            raise ValueError("The number of components cannot be less than 1")
        else:
            self.n_components = int(self.n_components)
        
        if self.init not in ['random', 'plus', 'kmeans', 'AF_KMC']:
            raise ValueError("Invalid value for 'init': %s "
                             "'init' should be in "
                             "['random', 'plus', 'kmeans', 'AF_KMC']"
                             % self.init)
    
    def _initialize(self,points_data,points_test=None):
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
        
        from .initializations import initialization_random
        from .initializations import initialization_plus_plus
        from .initializations import initialization_AF_KMC
        
        n_points,dim = points_data.shape
        
        #K-means++ initialization
        if (self.init == "random"):
            means = initialization_random(self.n_components,points_data)
            self.means = means
            self.log_weights = np.zeros(self.n_components) - np.log(self.n_components)
            self.iter = 0
        elif (self.init == "plus"):
            means = initialization_plus_plus(self.n_components,points_data)
            self.means = means
            self.log_weights = np.zeros(self.n_components) - np.log(self.n_components)
            self.iter = 0
        elif (self.init == "AF_KMC"):
            means = initialization_AF_KMC(self.n_components,points_data)
            self.means = means
            self.log_weights = np.zeros(self.n_components) - np.log(self.n_components)
            self.iter = 0
        elif (self.init == 'user'):
            pass
        else:
            raise ValueError("Invalid value for 'initialization': %s "
                                 "'initialization' should be in "
                                 "['random', 'plus','AF_KMC']"
                                  % self.init)
        self._is_initialized = True
        
        
    def _step_E(self,points):
        """
        This method assign a cluster number to each point by changing its last coordinate
        Ex : if P belongs to the first cluster, then P[-1] = 0.
        
        :param points: an array (n_points,dim)
        :return assignments: an array (n_components,dim)
        
        """
        n_points,_ = points.shape
        assignements = np.zeros((n_points,self.n_components))
        
        M = dist_matrix(points,self.means)
        for i in range(n_points):
            index_min = np.argmin(M[i]) #the cluster number of the ith point is index_min
            if (isinstance(index_min,np.int64)):
                assignements[i][index_min] = 1
            else: #Happens when two points are equally distant from a cluster mean
                assignements[i][index_min[0]] = 1
                
        return assignements
    
        
    def _step_M(self,points,assignements):
        """
        This method computes the new position of each means by minimizing the distortion
        
        Parameters
        ----------
        points : an array (n_points,dim)
        assignements : an array (n_components,dim)
            an array containing the responsibilities of the clusters
            
        """
        n_points,dim = points.shape
        
        for i in range(self.n_components):
            assignements_i = assignements[:,i:i+1]
            n_set = np.sum(assignements_i)
            idx_set,_ = np.where(assignements_i==1)
            sets = points[idx_set]
            if n_set > 0:
                self.means[i] = np.asarray(np.sum(sets, axis=0)/n_set)
                
            self.log_weights[i] = np.log(n_set + np.finfo(np.float64).eps)
    
    def score(self,points,assignements=None):
        """
        This method returns the distortion measurement at the end of the k_means.
        
        Parameters
        ----------
        points : an array (n_points,dim)
        assignements : an array (n_components,dim)
            an array containing the responsibilities of the clusters
        Returns
        -------
        distortion : (float)
        
        """
        if assignements is None:
            assignements = self.predict_assignements(points)
        
        if self._is_initialized:
            n_points,_ = points.shape
            distortion = 0
            for i in range(self.n_components):
                assignements_i = assignements[:,i:i+1]
                n_set = np.sum(assignements_i)
                idx_set,_ = np.where(assignements_i==1)
                sets = points[idx_set]
                if n_set != 0:
                    M = dist_matrix(sets,self.means[i].reshape(1,-1))
                    distortion += np.sum(M)
                
            return distortion

        else:
            raise Exception("The model is not initialized")

        
    def fit(self,points_data,points_test=None,n_iter_max=100,
            n_iter_fix=None,tol=0,saving=None,file_name='model',
            saving_iter=2):
        """The k-means algorithm
        
        Parameters
        ----------
        points_data : array (n_points,dim)
            A 2D array of points on which the model will be trained
            
        tol : float, defaults to 0
            The EM algorithm will stop when the difference between two steps 
            regarding the distortion is less or equal to tol.
            
        n_iter_max : int, defaults to 100
            number of iterations maximum that can be done
            
        saving_iter : int | defaults 2
            An int to know how often the model is saved (see saving below).
            
        file_name : str | defaults model
            The name of the file (including the path).
        
        Other Parameters
        ----------------
        points_test : array (n_points_bis,dim) | Optional
            A 2D array of points on which the model will be tested.
        
        n_iter_fix : int | Optional
            If not None, the algorithm will exactly do the number of iterations
            of n_iter_fix and stop.
            
        saving : str | Optional
            A string in ['log','linear']. In the following equations x is the parameter
            saving_iter (see above).
            
            * If 'log', the model will be saved for all iterations which verify :
                log(iter)/log(x) is an int
                
            * If 'linear' the model will be saved for all iterations which verify :
                iter/x is an int
                
        Returns
        -------
        None
        
        """
        n_points,_ = points_data.shape
        
        #Initialization
        if not self._is_initialized or self.init!='user':
            self._initialize(points_data,points_test)
            self.iter = 0        
            
        if saving is not None:
            f = h5py.File(file_name + '.h5', 'a')
            grp = f.create_group('best' + str(self.iter))
            self.write(grp)
            f.close()
        
        condition = _check_saving(saving,saving_iter)
        
        early_stopping = points_test is not None
        first_iter = True
        resume_iter = True
        
        dist_data, dist_test = 0,0
        
        #K-means beginning
        while resume_iter:
            
            assignements_data = self._step_E(points_data)
            dist_data_pre = dist_data
            if early_stopping:
                assignements_test = self._step_E(points_test)
                dist_test_pre = dist_test
            
            self._step_M(points_data,assignements_data)
            dist_data = self.score(points_data,assignements_data)
            if early_stopping:
                dist_test = self.score(points_test,assignements_test)
            
            self.iter+=1

            # Computation of resume_iter
            if n_iter_fix is not None:
                resume_iter = self.iter < n_iter_fix
            
            elif first_iter:
                first_iter = False
                
            elif self.iter > n_iter_max:
                resume_iter = False
                
            else:
                if early_stopping:
                    criterion = (dist_test_pre - dist_test)/len(points_test)
                else:
                    criterion = (dist_data_pre - dist_data)/n_points
                resume_iter = (criterion > tol)
                
            if not resume_iter and saving is not None:
                f = h5py.File(file_name + '.h5', 'a')
                grp = f.create_group('best' + str(self.iter))
                self.write(grp)
                f.close()
            
            elif condition(self.iter):
                f = h5py.File(file_name + '.h5', 'a')
                grp = f.create_group('iter' + str(self.iter))
                self.write(grp)
                f.close()

            
    def predict_assignements(self,points):
        """
        This function return the hard assignements of points once the model is
        fitted.
        
        """
    
        if self._is_initialized:
            assignements = self._step_E(points)
            return assignements

        else:
            raise Exception("The model is not initialized")
            

    def _get_parameters(self):
        return (self.log_weights, self.means)
    

    def _set_parameters(self, params,verbose=True):
        self.log_weights, self.means = params
        
        if self.n_components != len(self.means) and verbose:
            print('The number of components changed')
        self.n_components = len(self.means)


    def _limiting_model(self,points):
        
        n_points,dim = points.shape
        log_resp = self.predict_log_resp(points)
        _,n_components = log_resp.shape
    
        exist = np.zeros(n_components)
        
        for i in range(n_points):
            for j in range(n_components):
                if np.argmax(log_resp[i])==j:
                    exist[j] = 1
        

        idx_existing = np.where(exist==1)
        
        log_weights = self.log_weights[idx_existing]
        means = self.means[idx_existing]
                
        params = (log_weights, means)
        
        return params