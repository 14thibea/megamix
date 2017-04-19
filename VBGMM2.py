# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:37:08 2017

@author: Calixi
"""

import utils
import Initializations as Init

import pickle
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import os
import numpy as np
import scipy.special
from scipy.misc import logsumexp
#import sklearn_test

class VariationalGaussianMixture():

    def __init__(self, n_components=1,init="GMM",n_iter_max=100,alpha_0=1.0,\
                 beta_0=1.0,nu_0=None,tol=1e-4,patience=0):
        
        super(VariationalGaussianMixture, self).__init__()

        self.n_components = n_components
        self.init = init
        self.n_iter_max = n_iter_max
        self.tol = tol
        self.patience = patience
        
        self._alpha_0 = alpha_0
        self._beta_0 = beta_0
        self._nu_0 = nu_0

    def _check_parameters(self,points):
        
        _,dim = points.shape
        
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
        n_components,dim,_ = self.inv_prec.shape

        for i in range(self.n_components):
            prec = np.linalg.inv(self.inv_prec[i])
            chol = np.linalg.cholesky(prec)
            
            A_diag = np.sqrt(np.random.chisquare(self._nu[i] - np.arange(0,dim), size = dim))
            A = np.diag(A_diag)
            A[np.tri(dim,k=-1,dtype = bool)] = np.random.normal(size = (dim*(dim-1))//2)
            
            X = np.dot(chol,A)
            prec_estimated = np.dot(X,X.T)
            
            self.cov_estimated[i] = np.linalg.inv(prec_estimated)
        
            self.means_estimated[i] = np.random.multivariate_normal(self.means[i] , self.cov_estimated[i]/self._beta[i])
    
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
        self.inv_prec_init = Init.initialization_full_covariances(points_data,self.n_components) #TODO utiliser la variance du problème
        
        if (self.init == "random"):
            self.means = Init.initialization_random(points_data,self.n_components)
            self.inv_prec = np.tile(self.inv_prec_init, (self.n_components,1,1))
            self.log_det_inv_prec = np.linalg.det(self.inv_prec)
        elif(self.init == "plus"):
            self.means = Init.initialization_plus_plus(points_data,self.n_components)
            self.inv_prec = np.tile(self.inv_prec_init, (self.n_components,1,1))
            self.log_det_inv_prec = np.linalg.det(self.inv_prec)
        elif(self.init == "kmeans"):
            self.means = Init.initialization_k_means(points_data,self.n_components)
            self.inv_prec = np.tile(self.inv_prec_init, (self.n_components,1,1))
            self.log_det_inv_prec = np.linalg.det(self.inv_prec)
        elif(self.init == "GMM"):
            means,inv_prec = Init.initialization_GMM(self.n_components,points_data,points_test=points_test)
            self.means = means
            self.inv_prec = inv_prec
            self.log_det_inv_prec = np.linalg.det(inv_prec)
            
        if self._nu_0 == None:
            self._nu_0 = dim
        
        if self._nu_0 < dim:
            raise ValueError("nu_0 must be superior to the dimension of the"
                             "problem or the gamma function won't be defined"
                             % self._nu_0)
        
        self._alpha = self._alpha_0 * np.ones(self.n_components)
        self._beta = self._beta_0 * np.ones(self.n_components)
        self._nu = self._nu_0 * np.ones(self.n_components)
        
        self.cov_estimated = self.inv_prec / np.tile(self._nu, (dim,dim,1)).T
        self.means_estimated = self.means
        
        
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
            log_det_prec_i = digamma_sum + dim * np.log(2) - self.log_det_inv_prec[i] #/!\ Inverse
            
            points_centered = points - self.means[i]
            prec = np.linalg.inv(self.inv_prec[i])
            
            esp_cov_mean = dim/self._beta[i] + self._nu[i] * np.dot(points_centered,np.dot(prec,points_centered.T))
            esp_cov_mean = np.diagonal(esp_cov_mean)
             
            # Formula page 476 Bishop
            log_pho_i = log_weights_i + 0.5*log_det_prec_i - dim*0.5*np.log(2*np.pi) - 0.5*esp_cov_mean
            
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
            diff_weighted = diff * np.tile(resp[:,i:i+1], (1,dim))
            S[i] = 1/N[i] * np.dot(diff_weighted.T,diff)
        
        #Parameters update
        self._alpha = self._alpha_0 + N
        self._beta = self._beta_0 + N
        self._nu = self._nu_0 + N
        
        means = self._beta_0 * self.means_init + np.tile(N,(dim,1)).T * X_barre
        self.means = means * np.tile(np.reciprocal(self._beta), (dim,1)).T
        self.means_estimated = self.means
        
        for i in range(self.n_components):
            diff = X_barre[i] - self.means_init
            product = self._beta_0 * N[i]/(self._beta[i]) * np.outer(diff,diff)
            self.inv_prec[i] = (self.inv_prec_init + N[i] * S[i] + product)
            det_inv_prec = np.linalg.det(self.inv_prec[i])
            self.log_det_inv_prec[i] = np.log(det_inv_prec)
            self.cov_estimated[i] = self.inv_prec[i] / self._nu[i]
        
        self.log_prior_prob = logsumexp(log_resp, axis=0) - np.log(n_points)
    
    def lower_bound(self,points,log_resp):
        
        resp = np.exp(log_resp)
        n_points,dim = points.shape
        
        # Convenient statistics
        N = np.sum(resp,axis=0)                                          #Array (n_components,)
        X_barre = np.tile(1/N, (dim,1)).T * np.dot(resp.T,points)        #Array (n_components,dim)
        S = np.zeros((self.n_components,dim,dim))
        
        prec = np.linalg.inv(self.inv_prec)
        prec_init = np.linalg.inv(self.inv_prec_init)
        
        log_weights = np.zeros(self.n_components)
        log_det_prec = np.zeros(self.n_components)
        lower_bound = np.zeros(self.n_components)
        
        for i in range(self.n_components):
            
            log_weights[i] = scipy.special.psi(self._alpha[i]) - scipy.special.psi(np.sum(self._alpha))
            
            digamma_sum = 0
            for j in range(dim):
                digamma_sum += scipy.special.psi((self._nu[i] - j)/2)
            log_det_prec[i] = digamma_sum + dim * np.log(2) - self.log_det_inv_prec[i] #/!\ Inverse
            
            #First line
            lower_bound[i] = log_det_prec[i] - dim/self._beta[i] - self._nu[i]*np.trace(np.dot(S[i],prec[i]))
            diff = X_barre[i] - self.means[i]
            lower_bound[i] += -self._nu[i]*np.dot(diff,np.dot(prec[i],diff.T)) - dim*np.log(2*np.pi)
            lower_bound[i] *= 0.5 * N[i]
            
            #Second line
            lower_bound[i] += (self._alpha_0 - self._alpha[i]) * log_weights[i]
            lower_bound[i] += utils.log_B(prec_init,self._nu_0) - utils.log_B(prec[i],self._nu[i])
            
            resp_i = resp[:,i:i+1]
            log_resp_i = log_resp[:,i:i+1]
            
            lower_bound[i] += np.sum(resp_i) * log_weights[i] - np.sum(resp_i*log_resp_i)
            lower_bound[i] += 0.5 * (self._nu_0 - self._nu[i]) * log_det_prec[i]
            lower_bound[i] += dim*0.5*(np.log(self._beta_0) - np.log(self._beta[i]))
            lower_bound[i] += dim*0.5*(1 - self._beta_0/self._beta[i] + self._nu[i])
            
            #Third line without the last term which is not summed
            diff = self.means[i] - self.means_init
            lower_bound[i] += -0.5*self._beta_0*self._nu[i]*np.dot(diff,np.dot(prec[i],diff.T))
            lower_bound[i] += -0.5*self._nu[i]*np.trace(np.dot(self.inv_prec_init,prec[i]))
                
        result = np.sum(lower_bound)
        result += utils.log_C(self._alpha_0 * np.ones(self.n_components))- utils.log_C(self._alpha)
        
        return result
        
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
        
        
        lb = self.lower_bound(points,log_resp)
        
        couleurs = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        
        x_points = [[] for i in range(self.n_components)]
        y_points = [[] for i in range(self.n_components)]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title("lower bound = " + str(lb) + " iter = " + str(self.iter) + " k = " + str(self.n_components))
        
#        self._sampling_Normal_Wishart()


        for i in range(self.n_components):
            
            if self.log_prior_prob[i] > -3:
                
                col = couleurs[i%7]
                                     
                ell = utils.ellipses_multidimensional(self.cov_estimated[i],self.means[i])
                x_points[i] = [points[j][0] for j in range(n_points) if (np.argmax(log_resp[j])==i)]        
                y_points[i] = [points[j][1] for j in range(n_points) if (np.argmax(log_resp[j])==i)]
        
                ax.plot(x_points[i],y_points[i],col + 'o',alpha = 0.2)
                ax.plot(self.means[i][0],self.means[i][1],'kx')
                ax.add_artist(ell)
            
        
        titre = directory + '/figure_' + self.init + "_" + str(t)
        plt.savefig(titre)
        plt.close("all")
        
    def create_graph_lower_bound(self,t):
        
        dir_path = 'VBGMM/' + self.init + '/'
        directory = os.path.dirname(dir_path)
    
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)  
        
        n_iter = len(self.lower_bound_data)
        
        p1, = plt.plot(np.arange(n_iter),self.lower_bound_data,marker='x',label='data')
        if self.test_exists:
            p2, = plt.plot(np.arange(n_iter),self.lower_bound_test,marker='o',label='test')
        
        plt.title("lower bound evolution")
        plt.legend(handler_map={p1: HandlerLine2D(numpoints=4)})

        
        titre = directory + '/figure_' + self.init + "_" + str(t) + "_lb_evolution"
        plt.savefig(titre)
        plt.close("all")
    
    def predict(self,points_data,points_test=None,draw_graphs=False):
        """
        The EM algorithm
        
        @param points: an array (n_points,dim)
        @return resp: an array containing the responsibilities (n_points,n_components)
        """
        self._check_parameters(points_data)
        self._initialize(points_data,points_test)
        
        self.test_exists = not(points_test is None)
        self.lower_bound_data = []
        if not (points_test is None):
            self.lower_bound_test = []
        
        resume_iter = True
        first_iter = True
        self.iter = 0
        patience = 0
        
        # EM algorithm
        while resume_iter:
            
            log_resp_data = self.step_E(points_data)
            if self.test_exists:
                log_resp_test = self.step_E(points_test)
            
            self.step_M(points_data,log_resp_data)
            
            self.lower_bound_data.append(self.lower_bound(points_data,log_resp_data))
            if self.test_exists:
                self.lower_bound_test.append(self.lower_bound(points_test,log_resp_test))
            
            #Graphic part
            if draw_graphs:
                self.create_graph(points_data,log_resp_data,"data_iter" + str(self.iter))
                self.create_graph(points_test,log_resp_test,"test_iter" + str(self.iter))
            
            if first_iter:
                resume_iter = True
                first_iter = False
                
            elif self.test_exists:
#                if abs(self.lower_bound_test[self.iter] - self.lower_bound_test[self.iter-1]) < self.tol:
                if self.lower_bound_test[self.iter] <= self.lower_bound_test[self.iter-1]:
                    resume_iter = patience < self.patience
                    patience += 1
                    
            else:
#                if abs(self.lower_bound_data[self.iter] - self.lower_bound_data[self.iter-1]) < self.tol:
                if self.lower_bound_data[self.iter] <= self.lower_bound_data[self.iter-1]:
                    resume_iter = patience < self.patience
                    patience += 1
                    
            self.iter+=1
        
        if self.test_exists:
            return log_resp_data,log_resp_test
        else:
            return log_resp_data
    
if __name__ == '__main__':
    
    points_data = utils.read("D:/Mines/Cours/Stages/Stage_ENS/Code/data/EMGaussienne.data")
    points_test = utils.read("D:/Mines/Cours/Stages/Stage_ENS/Code/data/EMGaussienne.test")
    
    initializations = ["random","plus","kmeans","GMM"]
    
    path = 'D:/Mines/Cours/Stages/Stage_ENS/Code/data/data.pickle'
    with open(path, 'rb') as fh:
        data = pickle.load(fh)
        
    N=1500
        
    points = data['BUC']
    points_data = points[:N:]
    points_test = points_data
    
    _,dim = points_data.shape
    
    
#    for init in initializations:
#        print()
#        print(init)
#        print()
    init="GMM"

    for j in np.arange(1,11):
        print(j)
        print(">>predicting")
        VBGMM = VariationalGaussianMixture(j,init,alpha_0=1.0/j,beta_0=1.0,patience=10)
#        log_resp_data,log_resp_test = VBGMM.predict(points_data,points_test)
        log_resp_data = VBGMM.predict(points_data)
        print(">>creating graphs")
        VBGMM.create_graph(points_data,log_resp_data,str(j) + "_data")
#        VBGMM.create_graph(points_test,log_resp_test,str(j) + "_test")
        VBGMM.create_graph_lower_bound(j)
        print()
        