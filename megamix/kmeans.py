# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 14:20:27 2017

:author: Elina Thibeau-Sutre
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics.pairwise import euclidean_distances


def dist_matrix(points,means):
    """
    This method computes all the distances between the points and the actual means
    of the clusters, taking in account that the last coordinate is the cluster number
    
    @param points: an array (n_points,dim)
    @param means: an array containing the means of the clusters (n_components,dim)
    @return: a matrix which contains the distances between the ith point and the jth center
    (n_points,n_components)
    """

    dist_matrix = euclidean_distances(points,means)

    return dist_matrix

class Kmeans():

    def __init__(self,n_components=1,init="plus"):
        
        super(Kmeans, self).__init__()

        self.name = 'Kmeans'
        self.n_components = n_components
        self.init = init
        
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

    def _step_E(self,points):
        """
        This method assign a cluster number to each point by changing its last coordinate
        Ex : if P belongs to the first cluster, then P[-1] = 0.
        
        @param points: an array (n_points,dim)
        @param means: an array containing the means of the clusters (n_components,dim)
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
        
        @param points: an array (n_points,dim)
        @param means: an array containing the means of the clusters (n_components,dim)
        """
        n_points,dim = points.shape
        
        for i in range(self.n_components):
            sets = assignements[:,i:i+1]
            n_sets = np.sum(sets)
            sets = points * sets
            if n_sets > 0:
                self.means[i] = np.asarray(np.sum(sets, axis=0)/n_sets)
        
    def create_graph(self,points,directory,legend):
        """
        This method draws a 2D graph displaying the clusters and their means and saves it as a PNG file.
        If points have more than two coordinates, then it will be a projection including only the first coordinates.
        
        @param points: an array (n_points,dim)
        @param means: an array containing the means of the clusters (n_components,dim)
        @param t: the figure number
        """
        n_points,_ = points.shape
        
        assignements = self._step_E(points)
        
        x_points = [[] for i in range(self.n_components)]
        y_points = [[] for i in range(self.n_components)]
        
        dist = self.distortion(points,assignements)
        
        fig = plt.figure()
        plt.title("distortion = " + str(dist) + " n_iter = " + str(self.iter))
        ax = fig.add_subplot(111)
        
        for i in range(self.n_components):
            x_points[i] = [points[j][0] for j in range(n_points) if (assignements[j][i] == 1)]        
            y_points[i] = [points[j][1] for j in range(n_points) if (assignements[j][i] == 1)]
            
            if len(x_points[i]) != 0:
                ax.plot(x_points[i],y_points[i],'o')
                ax.plot(self.means[i][0],self.means[i][1],'kx')
        
        titre = directory + '/figure_' + str(legend) + ".png"
        plt.savefig(titre)
        plt.close("all")
    
    def distortion(self,points,assignements):
        """
        This method returns the distortion measurement at the end of the k_means.
        
        @param points: an array (n_points,dim)
        @param means: an array containing the means of the clusters (n_components,dim)
        @return: distortion measurement (float)
        """
        n_points,_ = points.shape
        distortion = 0
        for i in range(self.n_components):
            sets = [points[j] for j in range(n_points) if (assignements[j][i]==1)]
            sets = np.asarray(sets)
            if len(sets) != 0:
                M = dist_matrix(sets,self.means[i].reshape(1,-1))
                distortion += np.sum(M)
        return distortion
        
        
    def fit(self,points_data,points_test=None,n_iter_max=100,n_iter_fix=None,
            tol=1e-3,draw_graphs=False,directory=None):
        """
        This method returns an array of k points which will be used in order to
        initialize a k_means algorithm
        
        @param points: an array (n_points,dim)
        @param k: the number of clusters
        @param draw_graphs: a boolean to allow the algorithm to draw graphs if True
        @param initialization: a string in ['random','plus']
        @return: the means of the clusters (n_components,dim)
        """
        from .initializations import initialization_random
        from .initializations import initialization_plus_plus
        from .initializations import initialization_AF_KMC
        
        n_points,dim = points_data.shape
        
        if directory is None:
            directory = os.getcwd()
        
        #K-means++ initialization
        if (self.init == "random"):
            means = initialization_random(self.n_components,points_data)
        elif (self.init == "plus"):
            means = initialization_plus_plus(self.n_components,points_data)
        elif (self.init == "AF_KMC"):
            means = initialization_AF_KMC(self.n_components,points_data)
        else:
            raise ValueError("Invalid value for 'initialization': %s "
                                 "'initialization' should be in "
                                 "['random', 'plus','AF_KMC']"
                                  % self.init)
        self.means = means
        self.iter = 0
        self._is_initialized = True
        
        test_exists = points_test is not None
        first_iter = True
        resume_iter = True
        
        dist_data, dist_test = 0,0
        
        #K-means beginning
        while resume_iter:
            
            assignements_data = self._step_E(points_data)
            dist_data_pre = dist_data
            if test_exists:
                assignements_test = self._step_E(points_test)
                dist_test_pre = dist_test
            
            self._step_M(points_data,assignements_data)
            dist_data = self.distortion(points_data,assignements_data)
            if test_exists:
                dist_test = self.distortion(points_test,assignements_test)
            
            #Graphic part
            if draw_graphs:
                self.create_graph(points_data,directory,"data_iter" + str(self.iter))
                if test_exists:
                    self.create_graph(points_test,directory,"test_iter" + str(self.iter))
            
            # Computation of resume_iter
            if first_iter:
                first_iter = False
                
            elif n_iter_fix is not None:
                resume_iter = self.iter < n_iter_fix
                
            elif self.iter > n_iter_max:
                resume_iter = False
                
            else:
                if test_exists:
                    criterion = (dist_test_pre - dist_test)/len(points_test)
                else:
                    criterion = (dist_data_pre - dist_data)/n_points
                resume_iter = (criterion >= tol)
                
            self.iter+=1
            
            
    def predict_assignements(self,points):
    
        if self._is_initialized:
            assignements = self._step_E(points)
            return assignements

        else:
            raise Exception("The model is not initialized")