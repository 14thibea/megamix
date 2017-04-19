# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:34:50 2017

@author: Calixi
"""

import utils
import Initializations as Init

import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import os
from scipy.misc import logsumexp

class GaussianMixture():

    def __init__(self, n_components=1,covariance_type="full",init="kmeans"\
                 ,n_iter_max=100,tol=1e-3,patience=0,draw_graphs=False):
        
        super(GaussianMixture, self).__init__()

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.init = init
        
        self.tol = tol
        self.patience = patience
        self.n_iter_max = n_iter_max
        
        self.draw_graphs = draw_graphs

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
            
    def _initialize(self,points):
        """
        This method initializes the Gaussian Mixture Model by setting the values
        of the means and the covariances
        @param points: an array (n_points,dim)
        """
        
        n_points,dim = points.shape

        self.cov_init = Init.initialization_full_covariances(points,self.n_components)
        
        if (self.init == "random"):
            self.means = Init.initialization_random(points,self.n_components)
            self.cov = np.tile(self.cov_init, (self.n_components,1,1))
            self.log_det_cov = np.linalg.det(self.cov)
        elif(self.init == "plus"):
            self.means = Init.initialization_plus_plus(points,self.n_components)
            self.cov = np.tile(self.cov_init, (self.n_components,1,1))
            self.log_det_cov = np.linalg.det(self.cov)
        elif(self.init == "kmeans"):
            self.means = Init.initialization_k_means(points,self.n_components)
            self.cov = np.tile(self.cov_init, (self.n_components,1,1))
            self.log_det_cov = np.linalg.det(self.cov)
            
        self.log_prior_prob = - np.log(self.n_components) * np.ones(self.n_components)

    def log_normal_matrix(self,points):
        """
        This method computes the log of the density of probability of a normal law centered. Each line
        corresponds to a point from points.
        
        @param points: an array of points (n_points,dim)
        @param means: an array of k points which are the means of the clusters (n_components,dim)
        @param cov: an array of k arrays which are the covariance matrices (n_components,dim,dim)
        @return: an array containing the log of density of probability of a normal law centered (n_points,n_components)
        """
        
        nb_points,dim = points.shape
        
        log_normal_matrix = np.zeros((nb_points,self.n_components))
        
        for i in range(self.n_components):
            log_normal_probabilities = scipy.stats.multivariate_normal.logpdf(points,self.means[i],self.cov[i])
            log_normal_probabilities = np.reshape(log_normal_probabilities, (nb_points,1))
            log_normal_matrix[:,i:i+1] = log_normal_probabilities
                         
        return log_normal_matrix
    
    def full_covariance_matrix_f(self,points,assignements):
        """
        Compute the full covariance matrices
        """
        nb_points,dim = points.shape
        
        covariance = np.zeros((self.n_components,dim,dim))
        
        for i in range(self.n_components):
            assignements_i = assignements[:,i:i+1]       
            sum_assignement = np.sum(assignements_i)
            assignements_duplicated = np.tile(assignements_i, (1,dim))
            points_centered = points - self.means[i]
            points_centered_weighted = points_centered * assignements_duplicated
            covariance[i] = np.dot(points_centered_weighted.T,points_centered)/sum_assignement                  
        
        return covariance
    
    def spherical_covariance_matrix_f(self,points,assignements):
        """
        Compute the coefficients for the spherical covariances matrices
        """
        nb_points,dim = points.shape
        
        covariance = np.zeros(self.n_components)
    
        for i in range(self.n_components):
            assignements_i = assignements[:,i:i+1]
            
            sum_assignement = np.sum(assignements_i)
            assignements_duplicated = np.tile(assignements_i, (1,dim))
            points_centered = points - self.means[i]
            points_centered_weighted = points_centered * assignements_duplicated
            product = np.dot(points_centered_weighted,points_centered.T)
            covariance[i] = np.trace(product)/sum_assignement
        
        return covariance / dim
    
    def step_E(self,points):
        """
        This method returns the list of the soft assignements of each point to each cluster
        @param log_assignements: an array containing the log of soft assignements of every point (n_points,n_components)
        
        
        @param points: an array of points (n_points,dim)
        @return: log of the soft assignements of every point (n_points,n_components)
        """
        
        n_points = len(points)
        
        log_normal_matrix = self.log_normal_matrix(points)
        log_prior_duplicated = np.tile(self.log_prior_prob, (n_points,1))
        log_product = log_normal_matrix + log_prior_duplicated
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
        
        self.means = result_inter * sum_assignements_final
        
        #Phase 2:
        if self.covariance_type=="full":
            self.cov = self.full_covariance_matrix_f(points,assignements)
        elif self.covariance_type=="spherical":
            self.cov = self.spherical_covariance_matrix_f(points,assignements)
                        
        #Phase 3:
        self.log_prior_prob = logsumexp(log_assignements, axis=0) - np.log(n_points)
        
    def create_graph(self,points,log_assignements,t):
        """
        This method draws a 2D graph displaying the clusters and their means and saves it as a PNG file.
        If points have more than two coordinates, then it will be a projection including only the first coordinates.
        
        @param points: an array of points (n_points,dim)
        @param log_assignements: an array containing the log of soft assignements of every point (n_points,n_components)
        @param t: the figure number
        """
        
        n_points,dim = points.shape
    
        dir_path = 'GMM/' + self.covariance_type + '_covariance/'
        directory = os.path.dirname(dir_path)
    
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)  
        
        
        log_like = self.log_likelihood(points)
        
        couleurs = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        
        x_points = [[] for i in range(self.n_components)]
        y_points = [[] for i in range(self.n_components)]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title("log likelihood = " + str(log_like) + " k = " + str(self.n_components))
        
#        self._sampling_Normal_Wishart()


        for i in range(self.n_components):
            
            if self.log_prior_prob[i] > -4:
                
                col = couleurs[i%7]
                                     
                ell = utils.ellipses(self.cov[i],self.means[i])
                x_points[i] = [points[j][0] for j in range(n_points) if (np.argmax(log_assignements[j])==i)]        
                y_points[i] = [points[j][1] for j in range(n_points) if (np.argmax(log_assignements[j])==i)]
        
                ax.plot(x_points[i],y_points[i],col + 'o',alpha = 0.2)
                ax.plot(self.means[i][0],self.means[i][1],'kx')
                ax.add_artist(ell)
            
        
        titre = directory + '/figure_' + str(t)
        plt.savefig(titre)
        plt.close("all")
        
    def create_graph_log(self,t):
        
        dir_path = 'GMM/' + self.covariance_type + '_covariance/'
        directory = os.path.dirname(dir_path)
    
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)  
        
        n_iter = len(self.log_like_data)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title("log likelihood evolution")
        
        ax.plot(np.arange(n_iter),self.log_like_data,'bx--')
        ax.plot(np.arange(n_iter),self.log_like_test,'kx--')
            
        
        titre = directory + '/figure_' + str(t) + "_log_evolution"
        plt.savefig(titre)
        plt.close("all")
    
    def log_likelihood(self,points):
        """
        This method returns the log likelihood at the end of the k_means.
        
        @param points: an array of points (n_points,dim)
        @return: log likelihood measurement (float)
        """
        n_points = len(points)
        
        log_normal_matrix = self.log_normal_matrix(points)
        log_prior_duplicated = np.tile(self.log_prior_prob, (n_points,1))
        log_product = log_normal_matrix + log_prior_duplicated
        log_product = logsumexp(log_product,axis=1)
        return np.sum(log_product)
        
        
    def predict_log_assignements(self,points_data,points_test):
        """
        This method returns an array of k points which will be used in order to
        initialize a k_means algorithm
        
        @param points: an array of points (n_points,dim)
        @param n_components: the number of clusters (int)
        @param draw_graphs: a boolean to allow the algorithm to draw graphs if True
        @param initialization: a string in ['random', 'plus', 'kmeans']
        @return: the means  (n_components,dim)
                 the covariances  (n_components,dim,dim)
                 the log of the soft assignements of the points (n_points,n_components)
        """
        
        self._check_parameters()
        self._initialize(points_data)
        
        n_points,dim = points_data.shape
            
        self.log_like_data = []
        self.log_like_test = []
                                 
        resume_iter = True  
        first_iter = True
        t=0
        patience=0
    
        #EM algorithm
        while resume_iter:
            
            log_assignements_data = self.step_E(points_data)
            log_assignements_test = self.step_E(points_test)
            self.step_M(points_data,log_assignements_data)
            self.log_like_data.append(self.log_likelihood(points_data))
            self.log_like_test.append(self.log_likelihood(points_test))
            
            #Graphic part
            if self.draw_graphs:
                self.create_graph(points_data,log_assignements_data,"data_iter" + str(t))
                self.create_graph(points_test,log_assignements_test,"test_iter" + str(t))
            
            if first_iter:
                resume_iter = True
                first_iter = False
            elif abs(self.log_like_test[t] - self.log_like_test[t-1]) < self.tol:
#            elif self.log_like_test[t] < self.log_like_test[t-1] :
                resume_iter = patience < self.patience
                patience += 1
            
            t+=1
        
        return log_assignements_data,log_assignements_test
    
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
    points_test = points_data

    #GMM
    for i in range(5):
        print(i)
        GMM = GaussianMixture(i+1,covariance_type="full",patience=20,draw_graphs=False)
    
        print(">>predicting")
        log_assignements_data,log_assignements_test = GMM.predict_log_assignements(points_data,points_test)
        print(">>creating graph")
        GMM.create_graph(points_data,log_assignements_data,str(i))
    #   GMM.create_graph_log(str(i))
        print()