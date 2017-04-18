# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:59:05 2017

@author: Calixi
"""

import utils
import Initializations as Init

import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.special
from scipy.misc import logsumexp
#import sklearn_test

class VariationalGaussianMixture():

    def __init__(self, n_components=1,init="random",n_iter_max=100,alpha_0=1.0,\
                 beta_0=1.0,nu_0=1.5,tol=1e-3,patience=0):
        
        super(VariationalGaussianMixture, self).__init__()

        self.n_components = n_components
        self.init = init
        self.n_iter_max = n_iter_max
        self.tol = tol
        self.patience = patience
        
        self._alpha_0 = alpha_0
        self._beta_0 = beta_0
        self._nu_0 = nu_0

    def _check_parameters(self):
        
        if self.init not in ['random', 'plus', 'kmeans','GMM']:
            raise ValueError("Invalid value for 'init': %s "
                             "'init' should be in "
                             "['random', 'plus', 'kmeans','GMM']"
                             % self.init)
        
    def _sampling_Normal_Wishart(self):
        """
        Sampling mu and sigma from Normal-Wishart distribution.

        """
        # Create the matrix A of the Bartlett decomposition (cf Wikipedia)
        n_components,dim,_ = self.cov.shape
        chol = np.linalg.cholesky(self.cov)

        for i in range(self.n_components):
            prec = np.linalg.inv(self.cov[i])
            chol = np.linalg.cholesky(prec)
            
            A_diag = np.sqrt(np.random.chisquare(self._nu_0 - np.arange(0,dim), size = dim))
            A = np.diag(A_diag)
            A[np.tri(dim,k=-1,dtype = bool)] = np.random.normal(size = (dim*(dim-1))//2)
            
            X = np.dot(chol,A)
            prec_estimated = np.dot(X,X.T)/self._nu_0 #Normalisation
            
            cov_estimated = np.linalg.inv(prec_estimated)
            self.cov_estimated[i] = cov_estimated
        
            self.means_estimated[i] = np.random.multivariate_normal(self.means[i] , self.cov[i]/self._beta[i])
    
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
    
    def _initialize(self,points_data,points_test):
        """
        This method initializes the Variational Gaussian Mixture by setting the values
        of the means, the covariances and other parameters specific (alpha, beta, nu)
        @param points: an array (n_points,dim)
        @param alpha_0: a float which influences on the number of cluster kept
        @param beta_0: a float
        @param nu_0: a float
        """
        
        n_points,dim = points_data.shape
        
        self.log_prior_prob = - np.log(self.n_components) * np.ones(self.n_components)
        self.means_init = np.mean(points_data,axis=0)
        self.cov_init = Init.initialization_full_covariances(points_data,self.n_components)
        
        if (self.init == "random"):
            self.means = Init.initialization_random(points_data,self.n_components)
            self.cov = np.tile(self.cov_init, (k,1,1))
            self.log_det_cov = np.linalg.det(self.cov)
        elif(self.init == "plus"):
            self.means = Init.initialization_plus_plus(points_data,self.n_components)
            self.cov = np.tile(self.cov_init, (k,1,1))
            self.log_det_cov = np.linalg.det(self.cov)
        elif(self.init == "kmeans"):
            self.means = Init.initialization_k_means(points_data,self.n_components)
            self.cov = np.tile(self.cov_init, (k,1,1))
            self.log_det_cov = np.linalg.det(self.cov)
        elif(self.init == "GMM"):
            means,cov = Init.initialization_GMM(points_data,points_test,self.n_components)
            self.means = means
            self.cov = cov
            self.log_det_cov = np.linalg.det(cov)
        
        
        self._alpha = self._alpha_0 * np.ones(self.n_components)
        self._beta = self._beta_0 * np.ones(self.n_components)
        self._nu = self._nu_0 * np.ones(self.n_components)
        
        self.cov_estimated = np.zeros((self.n_components,dim,dim))
        self.means_estimated = np.zeros((self.n_components,dim))
        
        
    def step_E(self, points):
        """
        In this step the algorithm evaluates the responsibilities of each points in each cluster
        
        @param points: an array (n_points,dim)
        @return resp: an array containing the responsibilities (n_points,n_components)
        """
        
        n_points,dim = points.shape
        log_resp = np.zeros((n_points,self.n_components))
        
        for i in range(self.n_components):
            log_weights_i = scipy.special.psi(self._alpha[i]) - scipy.special.psi(np.sum(self._alpha))
            
            digamma_sum = 0
            for j in range(dim):
                digamma_sum += scipy.special.psi((self._nu[i] - j)/2)
            log_det_cov_i = digamma_sum + dim * np.log(2) - self.log_det_cov[i] #/!\ Inverse
            
            points_centered = points - self.means[i]
            cov_inv_i = np.linalg.inv(self.cov[i])
            cov_inv_i /= self._nu[i]
            # We divide by nu because of the previous normalization
            
            esp_cov_mean = dim/self._beta[i] + self._nu[i] * np.dot(points_centered,np.dot(cov_inv_i,points_centered.T))
            esp_cov_mean = np.diagonal(esp_cov_mean)
             
            # Formula page 476 Bishop
            log_pho_i = log_weights_i + 0.5*log_det_cov_i - dim*0.5*np.log(2*np.pi) - 0.5*esp_cov_mean
            
            log_pho_i = np.reshape(log_pho_i, (n_points,1))
            
            log_resp[:,i:i+1] = log_pho_i
        
        log_sum_pho = logsumexp(log_resp, axis=1)

        for i in range(n_points):
            log_resp[i] = log_resp[i] - log_sum_pho[i]
                    
        return log_resp
    
    def step_M(self,points,log_resp):
        """
        In this step the algorithm updates the values of the parameters (means, covariances,
        alpha, beta, nu).
        
        @param points: an array (n_points,dim)
        @param resp: an array containing the responsibilities (n_points,n_components)
        """
        
        n_points,dim = points.shape
        
        resp = np.exp(log_resp)
        
        # Convenient statistics
        N = np.sum(resp,axis=0)                                          #Array (n_components,)
        X_barre = np.tile(1/N, (dim,1)).T * np.dot(resp.T,points)        #Array (n_components,dim)
        S = np.zeros((self.n_components,dim,dim))                        #Array (n_components,dim,dim)
        for i in range(self.n_components):
            diff = points - X_barre[i]
            diff_weighthed = diff * np.tile(resp[:,i:i+1], (1,dim))
            S[i] = 1/N[i] * np.dot(diff_weighthed.T,diff)
        
        #Parameters update
        self._alpha = self._alpha_0 + N
        
        self._beta = self._beta_0 + N
        
        self._nu = self._nu_0 + N
        
        means = self._beta_0 * self.means_init + np.tile(N,(dim,1)).T * X_barre
        self.means = means * np.tile(np.reciprocal(self._beta), (dim,1)).T
        
        for i in range(self.n_components):
            diff = X_barre[i] - self.means_init
            product = self._beta_0 * N[i]/(self._beta[i]) * np.outer(diff,diff)
            self.cov[i] = (self.cov_init + N[i] * S[i] + product)
            #Normalisation not specified in Bishop's book but realised in scikit-learn
            self.cov[i] /= N[i]
            det_cov = np.linalg.det(self.cov[i])
            self.log_det_cov[i] = np.log(det_cov)
        
#        for i in range(self.n_components):
#            diff = X_barre[i] - self.means_init
#            product = self._beta_0 /(self._beta[i]) * np.outer(diff,diff)
#            self.cov[i] = (self.cov_init + S[i] + product)
#            self.log_det_cov[i] = np.log(np.linalg.det(self.cov[i]))
        
        self.log_prior_prob = logsumexp(log_resp, axis=0) - np.log(n_points)
        
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
        
    def create_graph(self,points,log_resp,t):
        """
        This method draws a 2D graph displaying the clusters and their means and saves it as a PNG file.
        If points have more than two coordinates, then it will be a projection including only the first coordinates.
        
        @param points: an array of points (n_points,dim)
        @param means: an array of k points which are the means of the clusters (n_components,dim)
        @param log_assignements: an array containing the log of soft assignements of every point (n_points,n_components)
        @full_covariance: a string in ['full','spherical']
        @param t: the figure number
        """
    
        n_points,dim = points.shape
    
        dir_path = 'VBGMM/' + self.init + '/'
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
        plt.title("log likelihood = " + str(log_like) + " iter = " + str(self.iter) + " k = " + str(self.n_components))
        
        self._sampling_Normal_Wishart()
        
        for i in range(self.n_components):
            
            if self.log_prior_prob[i] > -3:
                
                col = couleurs[i%7]
                                     
                ell = utils.ellipses(self.cov[i],self.means[i])
                x_points[i] = [points[j][0] for j in range(n_points) if (np.argmax(log_resp[j])==i)]        
                y_points[i] = [points[j][1] for j in range(n_points) if (np.argmax(log_resp[j])==i)]
        
                ax.plot(x_points[i],y_points[i],col + 'o',alpha = 0.2)
                ax.plot(self.means[i][0],self.means[i][1],'kx')
                ax.add_artist(ell)
            
        
        titre = directory + '/figure_' + self.init + "_" + str(t)
        plt.savefig(titre)
        plt.close("all")
        
    def predict(self,points_data,points_test,draw_graphs=False):
        """
        The EM algorithm
        
        @param points: an array (n_points,dim)
        @return resp: an array containing the responsibilities (n_points,n_components)
        """
        self._check_parameters()
        self._initialize(points_data,points_test)
        
        self.log_like_data = []
        self.log_like_test = []
        
        resume_iter = True
        first_iter = True
        self.iter = 0
        patience = 0
        
        # EM algorithm
        while resume_iter:
            
            log_assignements_data = self.step_E(points_data)
            log_assignements_test = self.step_E(points_test)
            self.step_M(points_data,log_assignements_data)
            self.log_like_data.append(self.log_likelihood(points_data))
            self.log_like_test.append(self.log_likelihood(points_test))
            
            #Graphic part
            if draw_graphs:
                self.create_graph(points_data,log_assignements_data,"data_iter" + str(self.iter))
                self.create_graph(points_test,log_assignements_test,"test_iter" + str(self.iter))
            
            if first_iter:
                resume_iter = True
                first_iter = False
            elif abs(self.log_like_test[self.iter] - self.log_like_test[self.iter-1]) < self.tol:
#            elif self.log_like_test[t] < self.log_like_test[t-1] :
                resume_iter = patience < self.patience
                patience += 1
            
            self.iter+=1
        
        return log_assignements_data,log_assignements_test
    
if __name__ == '__main__':
    
    k=4
    
    points_data = utils.read("D:/Mines/Cours/Stages/Stage_ENS/Code/data/EMGaussienne.data")
    points_test = utils.read("D:/Mines/Cours/Stages/Stage_ENS/Code/data/EMGaussienne.test")
    
    n_points,dim = points_data.shape
    
    initializations = ["random","plus","kmeans","GMM"]
    
#    for init in initializations:
    init = "GMM"
    print(init)
    print()
    
    for j in range(10):
        print(j)
        print(">>predicting")
        VBGMM = VariationalGaussianMixture(j+1,init,alpha_0=1.0,beta_0=1.0,nu_0=100)
        log_resp_data,log_resp_test = VBGMM.predict(points_data,points_test)
        print(">>creating graphs")
        VBGMM.create_graph(points_data,log_resp_data,str(j+1) + "_data")
        VBGMM.create_graph(points_test,log_resp_test,str(j+1) + "_test")
        print()
        