# -*- coding: utf-8 -*-
#
#Created on Mon Apr 10 14:20:27 2017
#
#author: Elina Thibeau-Sutre
#

import numpy as np
import os
from sklearn.metrics.pairwise import euclidean_distances
from megamix.batch.initializations import initialization_plus_plus

def dist_matrix(points,means):
    
    if len(points) > 1:
        dist_matrix = euclidean_distances(points,means)
    else:
        dist_matrix = np.linalg.norm(points-means,axis=1)

    return dist_matrix

class Kmeans():

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
    def __init__(self,n_components=1,n_jobs=1,kappa=1.0):
        
        super(Kmeans, self).__init__()

        self.name = 'Kmeans'
        self.n_components = n_components
        self.kappa = kappa
        self.n_jobs = n_jobs
        
        self._is_initialized = False
        self.iter = 0
        
        self._check_parameters()

    def _check_parameters(self):
        
        if self.n_components < 1:
            raise ValueError("The number of components cannot be less than 1")
        else:
            self.n_components = int(self.n_components)
            
        if self.kappa <= 0 or self.kappa > 1:
            raise ValueError("kappa must be in ]0,1]")
    
    def _initialize(self,points):
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
        n_points,dim = points.shape
        
        self.means = initialization_plus_plus(self.n_components,points)
        
        self.N = 1/self.n_components * np.ones(self.n_components)
        self.X =  self.means * self.N[:,np.newaxis]
        self.iter = self.n_components + 1
        
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
        
    def _step_M(self,point,assignement):
        """
        This method computes the new position of each means by minimizing the distortion
        
        Parameters
        ----------
        points : an array (n_points,dim)
        assignements : an array (n_components,dim)
            an array containing the responsibilities of the clusters
            
        """
        n_point,dim = point.shape

        # New sufficient statistics
        N = assignement.reshape(self.n_components)
        X = assignement.T * point
        
        # Sufficient statistics update
        gamma = 1/(self.iter**self.kappa)
        
        self.N = (1-gamma)*self.N + gamma*N
        self.X = (1-gamma)*self.X + gamma*X     
        
        # Parameter update
        self.means = self.X / self.N[:,np.newaxis]
    
    def distortion(self,points,assignements):
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

        
    def fit(self,points,directory=None,offset=None):
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
        
        Other Parameters
        ----------------
        points_test : array (n_points_bis,dim) | Optional
            A 2D array of points on which the model will be tested.
        
        n_iter_fix : int | Optional
            If not None, the algorithm will exactly do the number of iterations
            of n_iter_fix and stop.
            
        Returns
        -------
        None
        
        """
        n_points,dim = points.shape
        
        #K-means beginning
        if directory is None:
            directory = os.getcwd()
            
        if offset is None:
            offset = self.n_components
        
        self._initialize(points[:offset:])
        
        for i in range(self.n_components,n_points):
            point = points[i].reshape(1,dim)
            resp = self._step_E(point)
            self._step_M(point,resp)
            self.iter += 1
            
            
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