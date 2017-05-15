# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:13:09 2017

@author: Elina THIBEAU-SUTRE
"""

import utils

from abc import abstractmethod
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import numpy as np
import scipy.linalg
from sklearn import manifold

def _full_covariance_matrix(points,means,weights,log_assignements,reg_covar):
    """
    Compute the full covariance matrices
    """
    nb_points,dim = points.shape
    n_components = len(means)
    
    covariance = np.zeros((n_components,dim,dim))
    
    for i in range(n_components):
        log_assignements_i = log_assignements[:,i]
        
        # We use the square root of the assignement values because values are very
        # small : this ensure that the matrix will be symmetric
        sqrt_assignements_i = np.exp(0.5*log_assignements_i)
        
        points_centered = points - means[i]
        points_centered_weighted = points_centered * sqrt_assignements_i[:,np.newaxis]
        covariance[i] = np.dot(points_centered_weighted.T,points_centered_weighted)
        covariance[i] = covariance[i] / weights[i]
        
        covariance[i] += reg_covar * np.eye(dim)
    
    return covariance

def _spherical_covariance_matrix(points,means,weights,assignements,reg_covar):
    """
    Compute the coefficients for the spherical covariances matrices
    """
    n_points,dim = points.shape
    n_components = len(means)
    
    covariance = np.zeros(n_components)

    for i in range(n_components):
        assignements_i = assignements[:,i:i+1]
        sum_assignement = np.sum(assignements_i)
        sum_assignement += 10 * np.finfo(assignements.dtype).eps
        
        points_centered = points - means[i]
        points_centered_weighted = points_centered * assignements_i
        product = points_centered * points_centered_weighted
        covariance[i] = np.sum(product)/sum_assignement
        
        covariance[i] += reg_covar
    
    return covariance / dim

def _compute_precisions_chol(cov,covariance_type):
    
     if covariance_type in 'full':
        n_components, n_features, _ = cov.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(cov):
            try:
                cov_chol = scipy.linalg.cholesky(covariance, lower=True)
            except scipy.linalg.LinAlgError:
                raise ValueError(str(k) + "-th covariance matrix non positive definite")
            precisions_chol[k] = scipy.linalg.solve_triangular(cov_chol,
                                                               np.eye(n_features),
                                                               lower=True).T
     return precisions_chol

def _log_normal_matrix(points,means,cov,covariance_type):
    """
    This method computes the log of the density of probability of a normal law centered. Each line
    corresponds to a point from points.
    
    @param points: an array of points (n_points,dim)
    @param means: an array of k points which are the means of the clusters (n_components,dim)
    @param cov: an array of k arrays which are the covariance matrices (n_components,dim,dim)
    @return: an array containing the log of density of probability of a normal law centered (n_points,n_components)
    """
    n_points,dim = points.shape
    n_components,_ = means.shape
    
    
    if covariance_type == "full":
        precisions_chol = _compute_precisions_chol(cov,covariance_type)
        log_det_chol = np.log(np.linalg.det(precisions_chol))
        
        log_prob = np.empty((n_points,n_components))
        for k, (mu, prec_chol) in enumerate(zip(means,precisions_chol)):
            y = np.dot(points,prec_chol) - np.dot(mu,prec_chol)
            log_prob[:,k] = np.sum(np.square(y), axis=1)
            
    elif covariance_type == "spherical":
        precisions_chol = np.sqrt(np.reciprocal(cov))
        log_det_chol = dim * np.log(precisions_chol)
        
        log_prob = np.empty((n_points,n_components))
        for k, (mu, prec_chol) in enumerate(zip(means,precisions_chol)):
            y = prec_chol * (points - mu)
            log_prob[:,k] = np.sum(np.square(y), axis=1)
            
    return -.5 * (dim * np.log(2*np.pi) + log_prob) + log_det_chol


class BaseMixture():
    """
    Base class for mixture models.
    This abstract class specifies an interface for other mixture classes and
    provides basic common methods for mixture models.
    """

    def __init__(self, n_components=1,init="GMM",n_iter_max=1000,
                 tol=1e-3,patience=0,type_init='resp'):
        
        super(BaseMixture, self).__init__()

        self.n_components = n_components
        self.init = init
        self.type_init = type_init
        
        self.tol = tol
        self.patience = patience
        self.n_iter_max = n_iter_max
        
    def _check_common_parameters(self):
        
        if self.n_components < 1:
            raise ValueError("The number of components cannot be less than 1")
        else:
            self.n_components = int(self.n_components)
            
                        
        if self.type_init not in ['resp','mcw']:
            raise ValueError("Invalid value for 'type_init': %s "
                             "'type_init' should be in "
                             "['resp','mcw']"
                             % self.type_init)
            
    def _check_hyper_parameters(self,n_points,dim):
        """
        This function tests the hyperparameters of the VBGMM and the DBGMM
        """
        
        #Checking alpha_0
        if self._alpha_0 is None:
            self._alpha_0 = 1/self.n_components
        elif self._alpha_0 < 0:
            raise ValueError("alpha_0 must be positive")
        
        #Checking beta_0
        if self._beta_0 is None:
            self._beta_0 = 1.0
        
        #Checking nu_0
        if self._nu_0 is None:
            self._nu_0 = dim
        
        elif self._nu_0 < dim:
            raise ValueError("nu_0 must be more than the dimension of the"
                             "problem or the gamma function won't be defined")
            
    def _sampling_Normal_Wishart(self):
        """
        Sampling mu and sigma from Normal-Wishart distribution.
        This method may only be used in VBGMM and DPGMM

        """
        # Create the matrix A of the Bartlett decomposition (cf Wikipedia)
        n_components,dim,_ = self.inv_prec.shape

        cov_estimated = np.zeros((n_components,dim,dim))
        means_estimated = np.zeros((n_components,dim))

        for i in range(self.n_components):
            prec = np.linalg.inv(self.inv_prec[i])
            chol = np.linalg.cholesky(prec)
            
            A_diag = np.sqrt(np.random.chisquare(self._nu[i] - np.arange(0,dim), size = dim))
            A = np.diag(A_diag)
            A[np.tri(dim,k=-1,dtype = bool)] = np.random.normal(size = (dim*(dim-1))//2)
            
            X = np.dot(chol,A)
            prec_estimated = np.dot(X,X.T)
            
            cov_estimated[i] = np.linalg.inv(prec_estimated)
            means_estimated[i] = np.random.multivariate_normal(self.means[i] , self.cov_estimated[i]/self._beta[i])
    
        return means_estimated, cov_estimated
    
    @abstractmethod   
    def convergence_criterion(self,points,log_resp):
        """
        The convergence criterion is different for GMM and VBGMM/DPGMM :
            - in GMM the log likelihood is used
            - in VBGMM/DPGMM the lower bound of the log likelihood is used
        """
        pass
        
    def create_graph(self,points,log_resp,t):
        """
        This method draws a 2D graph displaying the clusters and their means and saves it as a PNG file.
        If points have more than two coordinates, then it will be a projection including only the first coordinates.
        
        @param points: an array of points (n_points,dim)
        @param means: an array of k points which are the means of the clusters (n_components,dim)
        @param log_assignements: an array containing the log of soft assignements of every point (n_points,n_components)
        @param t: the figure number
        """
    
        n_points,dim = points.shape
        
        plt.title("iter = " + str(self.iter) + " k = " + str(self.n_components))
        
        if self.covariance_type == "full":
            cov = self.cov
        elif self.covariance_type == "spherical":
            cov = np.asarray([np.eye(dim) * coeff for coeff in self.cov])
        
        couleurs = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        
        x_points = [[] for i in range(self.n_components)]
        y_points = [[] for i in range(self.n_components)]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        for i in range(self.n_components):
            
            if self.log_weights[i] > -10:
                
                col = couleurs[i%7]
                                     
                ell = utils.ellipses_multidimensional(cov[i],self.means[i])
                x_points[i] = [points[j][0] for j in range(n_points) if (np.argmax(log_resp[j])==i)]        
                y_points[i] = [points[j][1] for j in range(n_points) if (np.argmax(log_resp[j])==i)]
        
                ax.plot(x_points[i],y_points[i],col + 'o',alpha = 0.2)
                ax.plot(self.means[i][0],self.means[i][1],'kx')
                ax.add_artist(ell)
            
        
        directory = self.create_path()
        titre = directory + '/figure_' + self.init + "_" + str(t) + ".png"
        plt.savefig(titre)
        plt.close("all")
        
    def create_graph_convergence_criterion(self,t):
        """
        This method draws a graph displaying the evolution of the convergence criterion :
            - log likelihood for GMM (should never decrease)
            - the lower bound of the log likelihood for VBGMM/DPGMM (should never decrease)
        """
        
        plt.title("Convergence criterion evolution")
        
        p1, = plt.plot(np.arange(self.iter),self.convergence_criterion_data,marker='x',label='data')
        if self.test_exists:
            p3, = plt.plot(np.arange(self.iter),self.convergence_criterion_test,marker='o',label='test')
            
        plt.legend(handler_map={p1: HandlerLine2D(numpoints=4)})

        
        directory = self.create_path()
        titre = directory + '/figure_' + self.init + "_" + str(t) + "_cc_evolution.png"
        plt.savefig(titre)
        plt.close("all")
        
    def create_graph_weights(self,t):
        """
        This method draws an histogram illustrating the repartition of the weights of the clusters
        """
        
        plt.title("Weights of the clusters")
        
        plt.hist(np.exp(self.log_weights))
        
        directory = self.create_path()
        titre = directory + '/figure_' + self.init + "_" + str(t) + "_weights.png"
        plt.savefig(titre)
        plt.close("all")
        
    def create_graph_MDS(self,t):
        """
        This method displays the means of the clusters on a 2D graph in order to visualize
        the cosine distances between them
        (see scikit-learn doc on manifold.MDS for more information)
        """
        
        mds = manifold.MDS(2)
        norm = np.linalg.norm(self.means,axis=1)
        means_normed = self.means / norm[:,np.newaxis]
        
        coord = mds.fit_transform(means_normed)
        stress = mds.stress_
        
        plt.plot(coord.T[0],coord.T[1],'o')
        plt.title("MDS of the cosine distances between means, stress = " + str(stress))
        
        directory = self.create_path()
        titre = directory + '/figure_' + self.init + "_" + str(t) + "_means.png"
        plt.savefig(titre)
        plt.close("all")
        
    def create_graph_entropy(self,t):
        """
        This method draws an histogram illustrating the repartition of the
        entropy of the covariance matrices
        """
        
        plt.title('Entropy of the covariances')
        
        l,_ = np.linalg.eig(self.cov)
        norm = np.trace(self.cov.T)
        l = l/norm[:,np.newaxis]
        
        log_l = np.log(l)
        ent = - np.sum(log_l*l, axis=1)
        
        plt.hist(ent)
        
        directory = self.create_path()
        titre = directory + '/figure_' + self.init + "_" + str(t) + "_cov_entropy.png"
        plt.savefig(titre)
        plt.close("all")
       
    @abstractmethod
    def create_path(self):
        """
        Create a directory to store the graphs
        
        @return: the path of the directory (str)
        """
        pass
    
    def predict_log_assignements(self,points_data,points_test=None,draw_graphs=False):
        """
        The EM algorithm
        
        @param points: an array (n_points,dim)
        @return resp: an array containing the responsibilities (n_points,n_components)
        """
        
        self._initialize(points_data,points_test)
        
        self.test_exists = not(points_test is None)
        self.convergence_criterion_data = []
        if not (points_test is None):
            self.convergence_criterion_test = []
            
        resume_iter = True
        first_iter = True
        self.iter = 0
        patience = 0
            
        if draw_graphs:
            self.create_graph_weights("_init_")
            self.create_graph_entropy("_init_")

        # EM algorithm
        while resume_iter:
            
            log_prob_norm_data,log_resp_data = self.step_E(points_data)
            if self.test_exists:
                log_prob_norm_test,log_resp_test = self.step_E(points_test)
                
            self.step_M(points_data,log_resp_data)
            
            self.convergence_criterion_data.append(self.convergence_criterion(points_data,log_resp_data,log_prob_norm_data))
            if self.test_exists:
                self.convergence_criterion_test.append(self.convergence_criterion(points_test,log_resp_test,log_prob_norm_test))
                
            #Graphic part
            if draw_graphs:
#                self.create_graph(points_data,log_resp_data,"data_iter" + str(self.iter))
                self.create_graph_weights("_iter_" + str(self.iter))
                self.create_graph_entropy("_iter_" + str(self.iter))
#                if self.test_exists:
#                    self.create_graph(points_test,log_resp_test,"test_iter" + str(self.iter))
            
            if first_iter:
                resume_iter = True
                first_iter = False
                
            elif self.test_exists:
#                criterion = abs(self.convergence_criterion_test[self.iter] - self.convergence_criterion_test[self.iter-1])
                criterion = self.convergence_criterion_test[self.iter] - self.convergence_criterion_test[self.iter-1]
                criterion /= len(points_test)
                if criterion < self.tol:
#                if self.convergence_criterion_test[self.iter] <= self.convergence_criterion_test[self.iter-1]:
                    resume_iter = patience < self.patience
                    patience += 1
                    
            else:
                criterion = abs(self.convergence_criterion_data[self.iter] - self.convergence_criterion_data[self.iter-1])
                criterion /= len(points_data)
                if criterion < self.tol:
#                if self.convergence_criterion_data[self.iter] <= self.convergence_criterion_data[self.iter-1]:
                    resume_iter = patience < self.patience
                    patience += 1
                    
            self.iter+=1
        
        print("Number of iterations :", self.iter)
        
        if self.test_exists:
            return log_resp_data,log_resp_test
        else:
            return log_resp_data
    