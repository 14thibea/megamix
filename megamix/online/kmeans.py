# -*- coding: utf-8 -*-
#
#Created on Mon Apr 10 14:20:27 2017
#
#author: Elina Thibeau-Sutre
#
from megamix.batch.initializations import initialization_plus_plus
from megamix.online.base import _check_saving, BaseMixture

import numpy as np
import h5py

def dist_matrix(points,means):
    
    if len(points) > 1:
        XX = np.einsum('ij,ij->i', points, points)[:, np.newaxis]    # Size (n_points,1)
        squared_matrix = np.dot(points,means.T)                      # Size (n_points,n_components)
        YY = np.einsum('ij,ij->i', means, means)[np.newaxis, :]      # Size (1,n_components)
        
        squared_matrix *= -2
        squared_matrix += XX
        squared_matrix += YY
        np.maximum(squared_matrix, 0, out=squared_matrix)    
        
        return np.sqrt(squared_matrix, out=squared_matrix)
    
    else:
        dist_matrix = np.linalg.norm(points-means,axis=1)
        return dist_matrix

class Kmeans(BaseMixture):
    """Kmeans model.
    
    Parameters
    ----------
    
    n_components : int, defaults to 1.
        Number of clusters used.
    
    window : int, defaults to 1
        The number of points used at the same time in order to update the
        parameters.
    
    kappa : double, defaults to 1.0
        A coefficient in ]0.0,1.0] which give weight or not to the new points compared
        to the ones already used.
        
        * If kappa is nearly null, the new points have a big weight and the model may take a lot of time to stabilize.

        * If kappa = 1.0, the new points won't have a lot of weight and the model may not move enough from its initialization.
        
    Attributes
    ----------

    name : str
        The name of the method : 'Kmeans'
    
    log_weights : array of floats (n_components)
        Contains the logarithm of the mixing coefficients of the model.
        
    means : array of floats (n_components,dim)
        Contains the computed means of the model.
        
    N : array of floats (n_components,)
        The sufficient statistic updated during each iteration used to compute
        log_weights (this corresponds to the mixing coefficients).
    
    X: array of floats (n_components,dim)
        The sufficient statistic updated during each iteration used to compute
        the means.
    
    iter : int
        The number of points which have been used to compute the model.
    
    _is_initialized : bool
        Ensures that the model has been initialized before using other
        methods such as fit(), distortion() or predict_assignements().
    
    Raises
    ------
    ValueError : if the parameters are inconsistent, for example if the cluster number is negative, init_type is not in ['resp','mcw']...
    
    References
    ----------
    *Online but Accurate Inference for Latent Variable Models with Local Gibbs Sampling*, C. Dupuy & F. Bach
    'The remarkable k-means++ <https://normaldeviate.wordpress.com/2012/09/30/the-remarkable-k-means/>'_
 
    """
    def __init__(self,n_components=1,window=1,kappa=1.0):
        
        super(Kmeans, self).__init__()

        self.name = 'Kmeans'
        self.n_components = n_components
        self.kappa = kappa
        self.window = window
        self.init = 'usual'
        
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
    
    def initialize(self,points):
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
        
        if self.init == 'usual':
            self.means = initialization_plus_plus(self.n_components,points)
            #TODO improve this initialization
            self.log_weights = np.zeros(self.n_components) - np.log(self.n_components)
            self.iter = self.n_components + 1

        self.N = np.exp(self.log_weights)
        self.X = self.means * self.N[:,np.newaxis]
        
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

        # New sufficient statistics
        N = assignements.sum(axis=0)
        N /= n_points
        
        X = np.dot(assignements.T,points)
        X /= n_points
        
        # Sufficient statistics update
        gamma = 1/((self.iter + n_points//2)**self.kappa)
        
        self.N = (1-gamma)*self.N + gamma*N
        self.X = (1-gamma)*self.X + gamma*X     
        
        # Parameter update
        self.means = self.X / self.N[:,np.newaxis]
        self.log_weights = np.log(self.N)
    
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
        
        n_points,_ = points.shape
        if assignements is None:
            assignements = self.predict_assignements(points)
            
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
        
    def fit(self,points,saving=None,file_name='model',
            saving_iter=2):
        """The k-means algorithm
        
        Parameters
        ----------
        points : array (n_points,dim)
            A 2D array of points on which the model will be trained.
            
        saving_iter : int | defaults 2
            An int to know how often the model is saved (see saving below).
            
        file_name : str | defaults model
            The name of the file (including the path).
        
        Other Parameters
        ----------------
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
        n_points,dim = points.shape
        
        condition = _check_saving(saving,saving_iter)
            
        if self._is_initialized:
            for i in range(n_points//self.window):
                point = points[i*self.window:(i+1)*self.window:]
                resp = self._step_E(point)
                self._step_M(point,resp)
                self.iter += self.window
                
                if condition(i+1):
                    f = h5py.File(file_name + '.h5', 'a')
                    grp = f.create_group('iter' + str(self.iter))
                    self.write(grp)
                    f.close()
        else:
            raise ValueError('The model is not initialized')
            
            
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
            
            
    # To be consistent with the cython version
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
            return self.iter
        elif name=='window':
            return self.window
        elif name=='kappa':
            return self.kappa
        elif name=='name':
            return self.name