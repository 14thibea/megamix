# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:34:50 2017

@author: Calixi
"""

import utils
import Initializations as Init

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import os
from scipy.misc import logsumexp

def log_normal_matrix_f(points,means,cov):
    """
    This method computes the log of the density of probability of a normal law centered. Each line
    corresponds to a point from points.
    
    @param points: an array of points (n_points,dim)
    @param means: an array of k points which are the means of the clusters (n_components,dim)
    @param cov: an array of k arrays which are the covariance matrices (n_components,dim,dim)
    @return: an array containing the log of density of probability of a normal law centered (n_points,n_components)
    """
    
    nb_points,dim = points.shape
    k = len(means)
    
    log_normal_matrix = np.zeros((nb_points,k))
    
    for i in range(k):
        log_normal_probabilities = scipy.stats.multivariate_normal.logpdf(points,means[i],cov[i])
        log_normal_probabilities = np.reshape(log_normal_probabilities, (nb_points,1))
        log_normal_matrix[:,i:i+1] = log_normal_probabilities
                     
    return log_normal_matrix

def full_covariance_matrix_f(points,means,assignements):
    nb_points,dim = points.shape
    k = len(means)
    
    covariance = np.zeros((k,dim,dim))
    
    for i in range(k):
        assignements_i = assignements[:,i:i+1]       
        sum_assignement = np.sum(assignements_i)
        assignements_duplicated = np.tile(assignements_i, (1,dim))
        points_centered = points - means[i]
        points_centered_weighted = points_centered * assignements_duplicated
        covariance[i] = np.dot(points_centered_weighted.T,points_centered)/sum_assignement                  
    
    return covariance

def spherical_covariance_matrix_f(points,means,assignements):
    nb_points,dim = points.shape
    k = len(means)
    
    covariance = np.zeros(k)

    for i in range(k):
        assignements_i = assignements[:,i:i+1]
        
        sum_assignement = np.sum(assignements_i)
        assignements_duplicated = np.tile(assignements_i, (1,dim))
        points_centered = points - means[i]
        points_centered_weighted = points_centered * assignements_duplicated
        product = np.dot(points_centered_weighted,points_centered.T)
        covariance[i] = np.trace(product)/sum_assignement
    
    return covariance / dim

def step_E(points,means,cov,log_prior_prob):
    """
    This method returns the list of the soft assignements of each point to each cluster
    @param log_assignements: an array containing the log of soft assignements of every point (n_points,n_components)
    
    
    @param points: an array of points (n_points,dim)
    @param means: an array of k points which are the means of the clusters (n_components,dim)
    @param cov: an array of k arrays which are the covariance matrices (n_components,dim,dim)
    @param log_prior_prob: an array of k numbers which are the log of prior probabilities (n_components,)
    @return: log of the soft assignements of every point (n_points,n_components)
    """
    
    k=len(means)
    n_points = len(points)
    
    log_normal_matrix = log_normal_matrix_f(points,means,cov)
    log_prior_duplicated = np.tile(log_prior_prob, (n_points,1))
    log_product = log_normal_matrix + log_prior_duplicated
    log_product_sum = np.tile(logsumexp(log_product,axis=1),(k,1))
    
    return log_product - log_product_sum.T
  
def step_M(points,log_assignements,covariance_type):
    """
    This method computes the new position of each mean and each covariance matrix
    
    @param points: an array of points (n_points,dim)
    @param log_assignements: an array containing the log of soft assignements of every point (n_points,n_components)
    @covariance_type: a string in ['full','spherical']
    @return: the means  (n_components,dim)
             the covariances  (n_components,dim,dim)
             the log of the prior probabilities of the clusters (n_components)
    """
    nb_points = float(len(points))
    dim = len(points[0])
    
    assignements = np.exp(log_assignements)
    
    #Phase 1:
    result_inter = np.dot(assignements.T,points)
    sum_assignements = np.sum(assignements,axis=0)
    sum_assignements_final = np.reciprocal(np.tile(sum_assignements, (dim,1)).T)
    
    means = result_inter * sum_assignements_final
    
    #Phase 2:
    if covariance_type=="full":
        cov = full_covariance_matrix_f(points,means,assignements)
#        list_cov = np.asarray([covariance_matrix(list_points,list_means[i],np.transpose(list_soft_assignement)[i]) for i in range(k)])
    elif covariance_type=="spherical":
        cov = spherical_covariance_matrix_f(points,means,assignements)
                    
    #Phase 3:
    log_prior_prob = 1/nb_points * logsumexp(log_assignements, axis=0)
    
    return means,cov,log_prior_prob
    
def create_graph(points,means,cov,log_assignements,covariance_type,t):
    """
    This method draws a 2D graph displaying the clusters and their means and saves it as a PNG file.
    If points have more than two coordinates, then it will be a projection including only the first coordinates.
    
    @param points: an array of points (n_points,dim)
    @param means: an array of k points which are the means of the clusters (n_components,dim)
    @param log_assignements: an array containing the log of soft assignements of every point (n_points,n_components)
    @full_covariance: a string in ['full','spherical']
    @param t: the figure number
    """
    
    k=len(means)
    nb_points,dim = points.shape

    dir_path = 'GMM/' + covariance_type + '_covariance/'
    directory = os.path.dirname(dir_path)

    try:
        os.stat(directory)
    except:
        os.mkdir(directory)  
    
    
    log_like = log_likelihood(points,means,cov,log_assignements)
    
    x_points = [[] for i in range(k)]
    y_points = [[] for i in range(k)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111,aspect='equal')
    plt.title("log likelihood = " + str(log_like))
    
    
    for i in range(k):
        if covariance_type=="full":
            covariance = cov[i]
        elif covariance_type=="spherical":
            covariance = cov[i]*np.eye(dim,dim)
            
        ell = utils.ellipses(covariance,means[i])
        ax.add_artist(ell)
        x_points[i] = [points[j][0] for j in range(nb_points) if (log_assignements[j][i]>np.log(0.7))]        
        y_points[i] = [points[j][1] for j in range(nb_points) if (log_assignements[j][i]>np.log(0.7))]

        ax.plot(x_points[i],y_points[i],'o',alpha = 0.2)
        ax.plot(means[i][0],means[i][1],'kx')
        
    
    titre = directory + '/figure_' + covariance_type + "_" + str(t)
    plt.savefig(titre)
    plt.close("all")
    
def log_likelihood(points,means,cov,log_assignements):
    """
    This method returns the log likelihood at the end of the k_means.
    
    @param points: an array of points (n_points,dim)
    @param means: an array of k points which are the means of the clusters (n_components,dim)
    @param cov: an array of k arrays which are the covariance matrices (n_components,dim,dim)
    @param log_assignements: an array containing the log of soft assignements of every point (n_points,n_components)
    @return: log likelihood measurement (float)
    """
    n_points = len(points)

    log_prior_prob = logsumexp(log_assignements, axis=0) - np.log(n_points)
    
    log_normal_matrix = log_normal_matrix_f(points,means,cov)
    log_prior_duplicated = np.tile(log_prior_prob, (n_points,1))
    log_product = log_normal_matrix + log_prior_duplicated
    log_product = logsumexp(log_product,axis=1) / np.log(10) #We use base log in base 10
    return np.sum(log_product)
    
    
def GMM(points,n_components,draw_graphs=False,initialization="kmeans",epsilon=1e-3,covariance_type="full"):
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
    
    #Means initialization
    if (initialization == "random"):
        means = Init.initialization_random(points,n_components)
    elif (initialization == "plus"):
        means = Init.initialization_plus_plus(points,n_components)
    elif (initialization == "kmeans"):
        means = Init.initialization_k_means(points,n_components)
    else:
        raise ValueError("Invalid value for 'initialization': %s "
                             "'initialization' should be in "
                             "['random', 'plus', 'kmeans']"
                              % initialization)

    
    n_points,dim = points.shape
    
    if (covariance_type=="full"):
        cov = Init.initialization_full_covariances(points,n_components)
    elif (covariance_type=="spherical"):
        cov = Init.initialization_spherical_covariances(points,n_components)
    
    log_prior_prob = - np.ones(n_components) * np.log(n_components)
                             
    resume_iter = True  
    t=0       
    
    #K-means beginning
    while resume_iter:
        
        log_assignements = step_E(points,means,cov,log_prior_prob)
        log_like_pre = log_likelihood(points,means,cov,log_assignements)
        means,cov,log_prior_prob = step_M(points,log_assignements,covariance_type)
        log_like = log_likelihood(points,means,cov,log_assignements)
        
        #Graphic part
        if draw_graphs:
            create_graph(points,means,cov,log_assignements,covariance_type,t)
        
        t+=1
        
        resume_iter = abs(log_like - log_like_pre) > epsilon
    
    return means,cov,log_assignements
    
if __name__ == '__main__':
    
    #Lecture du fichier
    points = utils.read("D:/Mines/Cours/Stages/Stage_ENS/Code/Problem/data/EMGaussienne.data")
    covariance_type = "full"
    
    k=4
    
    #GMM
    for i in range(20):
        means,cov,log_assignements = GMM(points,k,draw_graphs=False,initialization="kmeans",covariance_type=covariance_type)
        create_graph(points,means,cov,log_assignements,covariance_type,i)