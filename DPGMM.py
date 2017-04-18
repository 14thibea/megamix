# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 09:52:51 2017

@author: Calixi
"""

import utils
import Initializations as Init
import GMM3

import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.special import psi
from scipy.misc import logsumexp
#import sklearn_test

class DirichletProcessGaussianMixture():

    def __init__(self, n_components=1,init="random",n_iter_max=100,alpha_0=1.0,\
                 beta_0=1.0,nu_0=1.5,tol=1e-3): #, covariance_type='full'):
        
        super(DirichletProcessGaussianMixture, self).__init__()

        self.n_components = n_components
        self.init = init
        self.n_iter_max = n_iter_max
        self.tol = tol
        
        self._alpha_0 = np.asarray([1,alpha_0])
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
    
    def _initialize(self,points):
        """
        This method initializes the Variational Gaussian Mixture by setting the values
        of the means, the covariances and other parameters specific (alpha, beta, nu)
        @param points: an array (n_points,dim)
        @param alpha_0: a float which influences on the number of cluster kept
        @param beta_0: a float
        @param nu_0: a float
        """
        
        n_points,dim = points.shape
        
        self.means_init = np.zeros(dim) #TODO Better initialization
        self.cov_init = Init.initialization_full_covariances(points,self.n_components)
        
        if (self.init == "random"):
            self.means = Init.initialization_random(points,self.n_components)
            self.cov = np.tile(self.cov_init, (k,1,1))
            self.log_det_cov = np.linalg.det(self.cov)
        elif(self.init == "plus"):
            self.means = Init.initialization_plus_plus(points,self.n_components)
            self.cov = np.tile(self.cov_init, (k,1,1))
            self.log_det_cov = np.linalg.det(self.cov)
        elif(self.init == "kmeans"):
            self.means = Init.initialization_k_means(points,self.n_components)
            self.cov = np.tile(self.cov_init, (k,1,1))
            self.log_det_cov = np.linalg.det(self.cov)
        elif(self.init == "GMM"):
            means,cov = Init.initialization_GMM(points,self.n_components)
            self.means = means
            self.cov = cov
            self.log_det_cov = np.linalg.det(cov)
        
        
        self._alpha = np.tile(self._alpha_0, (self.n_components,1))
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
        log_weights = np.zeros(self.n_components)
        
        for i in range(self.n_components):
            if i==0:
                log_weights[i] = psi(self._alpha[i][0]) - psi(np.sum(self._alpha[i]))
            else:
                log_weights[i] = psi(self._alpha[i][0]) - psi(np.sum(self._alpha[i]))
                log_weights[i] += log_weights[i-1] + psi(self._alpha[i-1][1]) - psi(self._alpha[i-1][0])
            
            print(log_weights[i])
            digamma_sum = 0
            for j in range(dim):
                digamma_sum += psi((self._nu[i] - j)/2)
            log_det_cov_i = digamma_sum + dim * np.log(2) - self.log_det_cov[i] #/!\ Inverse
            
            points_centered = points - self.means[i]
            cov_inv_i = np.linalg.inv(self.cov[i])
            esp_cov_mean = dim/self._beta[i] + self._nu[i] * np.dot(points_centered,np.dot(cov_inv_i,points_centered.T))
            esp_cov_mean = np.diagonal(esp_cov_mean)
             
            # Formula page 476 Bishop
            log_pho_i = log_weights[i] + 0.5*log_det_cov_i - dim*0.5*np.log(2*np.pi) - 0.5*esp_cov_mean
            
            log_pho_i = np.reshape(log_pho_i, (n_points,1))
            
            log_resp[:,i:i+1] = log_pho_i
        
        log_sum_pho = logsumexp(log_resp, axis=1)
        print()
        
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
        for i in range(self.n_components):
            sum_N = np.sum(N[i+1::])
            self._alpha[i] = self._alpha_0 + np.asarray([N[i],sum_N])
        
        self._beta = self._beta_0 + N
        
        self._nu = self._nu_0 + N
        
        means = self._beta_0 * self.means_init + np.tile(N,(dim,1)).T * X_barre
        self.means = means * np.tile(np.reciprocal(self._beta), (dim,1)).T
        
        for i in range(self.n_components):
            diff = X_barre[i] - self.means_init
            product = self._beta_0 * N[i]/(self._beta[i]) * np.outer(diff,diff)
            self.cov[i] = (self.cov_init + N[i] * S[i] + product)
            self.log_det_cov[i] = np.log(np.linalg.det(self.cov[i]))
    
    def draw_graph(self,k,log_resp,points,t):
    
        n_points = len(points)
        couleurs = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        
        log_like = GMM3.log_likelihood(points,self.means,self.cov,log_resp)
        
        resp = np.exp(log_resp)
        
        fig = plt.figure()
        plt.title("log likelihood = " + str(log_like) + " iter = " + str(self.iter))
        ax = fig.add_subplot(111)
        
        
        for j in range(k):
            for i in range(n_points):        
                ax.plot(points[i][0],points[i][1],couleurs[j]+'o',alpha = resp[i][j]/5)
            ax.plot(self.means[j][0],self.means[j][1],'kx')
            ell = utils.ellipses(self.cov[j],self.means[j])
            ax.add_artist(ell)
            print("cluster " + str(j) + " finished")
        print()
            
        dir_path = 'DPGMM/' + self.init + '/'
        directory = os.path.dirname(dir_path)
    
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)  
        
        titre = directory + '/figure_' + self.init + "_k_" + str(k) + "_" + str(t)
                
        plt.savefig(titre)
        plt.close("all")
        
    def predict(self, points, draw_graphs=False):
        """
        The EM algorithm
        
        @param points: an array (n_points,dim)
        @return resp: an array containing the responsibilities (n_points,n_components)
        """
        self._check_parameters()
        self._initialize(points)
        
        resume_iter = True
        self.iter = 0
        
        while resume_iter:
            
            log_resp = self.step_E(points)
            log_like_pre = GMM3.log_likelihood(points,self.means,self.cov,log_resp)
            self.step_M(points,log_resp)
            log_like = GMM3.log_likelihood(points,self.means,self.cov,log_resp)
            self._sampling_Normal_Wishart()
            
            if draw_graphs == True:
                self.draw_graph(self.n_components,log_resp,points,self.iter)
            
            self.iter+=1
            
            resume_iter = (self.iter > self.n_iter_max) or abs(log_like - log_like_pre) > self.tol
        
        return log_resp
    
if __name__ == '__main__':
    
    k=7
    
    points = utils.read("D:/Mines/Cours/Stages/Stage_ENS/Code/data/EMGaussienne.data")
    n_points = len(points)
    
    initializations = ["random","plus","kmeans","GMM"]
    
#    for init in initializations:
    init = "kmeans"
    DPGMM = DirichletProcessGaussianMixture(k,init,alpha_0=0.01,beta_0=1.0,nu_0=10)
    print(init)
    print()
    
    for j in range(20):
        log_resp = DPGMM.predict(points)
        DPGMM.draw_graph(k,log_resp,points,j)