# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:59:05 2017

@author: Calixi
"""

import CSVreader
import Initializations as Init
import matplotlib.pyplot as plt
#from matplotlib.patches import Ellipse
import numpy as np
import scipy.special
import sklearn_test

class VariationalGaussianMixture():

    def __init__(self, n_components=1): #, covariance_type='full', tol=1e-3):
        
        super(VariationalGaussianMixture, self).__init__()

        self.n_components = n_components
    
    def _initialize(self, points, alpha_0, beta_0, nu_0):
        
        n_points,dim = points.shape
        
        self.means_init = np.zeros(dim) #TODO Better initialization
        self.means = Init.initialization_k_means(points,k)
        self.cov_init = np.eye(dim,dim) #TODO Better initialization
        self.cov = np.tile(np.eye(dim,dim), (self.n_components,1,1))
        self.log_det_cov = np.zeros(self.n_components)
        self._alpha_0 = alpha_0
        self._alpha = alpha_0 * np.ones(self.n_components)
        self._beta_0 = beta_0
        self._beta = beta_0 * np.ones(self.n_components)
        self._nu_0 = nu_0
        self._nu = nu_0 * np.ones(self.n_components)
        self.resp = np.zeros((n_points,self.n_components))
        
        points = points[:,0:-1] 
        
    def step_E(self, points):
        
        n_points,dim = points.shape
        resp = np.zeros((n_points,self.n_components))
        
        for i in range(self.n_components):
            log_pi_i = scipy.special.psi(self._alpha[i]) - scipy.special.psi(self.n_components * self._alpha_0 + n_points)
            
            digamma_sum = 0
            for j in range(dim):
                digamma_sum += scipy.special.psi((self._nu[i] - j)/2)
            log_det_cov_i = digamma_sum + dim * np.log(2) + self.log_det_cov[i]
            
            points_centered = points - self.means[i]
            cov_inv_i = np.linalg.inv(self.cov[i])
            esp_cov_mean = dim/self._beta[i] + self._nu[i] * np.dot(points_centered,np.dot(cov_inv_i,points_centered.T))
            esp_cov_mean = np.diagonal(esp_cov_mean)
             
            # Formula page 476 Bishop
            log_pho_i = log_pi_i + 0.5*log_det_cov_i - dim*0.5*np.log(2*np.pi) - 0.5*esp_cov_mean
            
            pho_i = np.exp(log_pho_i)
            pho_i = np.reshape(pho_i, (n_points,1))
            
            resp[:,i:i+1] = pho_i
        
        sum_pho = np.sum(resp, axis=1)

        for i in range(n_points):
            resp[i] = resp[i] / sum_pho[i]
                    
        return resp
    
    def step_M(self,points,resp):
        
        n_points,dim = points.shape
        
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
        print(self._alpha)
        
        self._beta = self._beta_0 + N
        
        self._nu = self._nu_0 + N + 1
        
        means = self._beta_0*self.means_init + np.tile(N,(dim,1)).T * X_barre
        self.means = means * np.tile(np.reciprocal(self._beta), (dim,1)).T
        
        for i in range(self.n_components):
            diff = X_barre[i] - self.means_init
            diff = np.reshape(diff, (dim,1))
            product = self._beta_0*N[i]/(self._beta_0 + N[i]) * np.dot(diff,diff.T)
            self.cov[i] = self.cov_init + N[i]*S[i] + product
            
    
if __name__ == '__main__':
    
    k=4
    
    points = CSVreader.read("D:/Mines/Cours/Stages/Stage_ENS/Code/Problem/EMGaussienne.data")
    points_init = np.copy(points)
    
    n_points = len(points)
    
    VBGMM = VariationalGaussianMixture(k)
    VBGMM._initialize(points,0.01,2.0,0.5)
    
    for i in range(20):
        resp = VBGMM.step_E(points)
        VBGMM.step_M(points,resp)
        
        labels = np.zeros(n_points)
        for j in range(n_points):
            labels[j] = np.argmax(resp[j])
        
        sklearn_test.draw(k,labels,points,i)
    