# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 11:31:51 2017

@author: Calixi
"""

import CSVreader
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import math

#def density_norm(point,mean,cov):
#    """
#    This method computes the density of probability of a normal law centered on mean
#    and with covariance point
#    
#    @param point: a nD array
#    @param mean: a nD array
#    @param cov: a nxn array, the covariance matrix
#    @return: the density of probability
#    """
#    dim = float(len(point))
#    point_centered = point - mean
#    det = np.linalg.det(cov)
#    scalar = (2*math.pi)**(dim/2) * math.sqrt(det)
#    product = np.dot(np.dot(point_centered.T,np.linalg.inv(cov)),point_centered)
#    return 1.0/scalar * math.exp(-0.5 * product)

#def covariance_matrix(list_points,mean,soft_assignements):
#    dim = len(list_points[0])
#    nb_points = len(list_points)
#    mean_matrix = np.tile(mean,(nb_points,1))
#    points_centered = list_points - mean_matrix
#    sum_soft_assignements = np.sum(soft_assignements)
#    soft_assignements_rep = (np.tile(soft_assignements,(dim,1))).T
#    return 1/sum_soft_assignements * np.dot(np.transpose(soft_assignements_rep*points_centered),points_centered)

def normal_matrix_f(points,means,cov,full_covariance):
    """
    This method computes the density of probability of a normal law centered. Each line
    corresponds to a point from points.
    
    @param points: an array (nb_points,dim)
    @param means: an array containing the k means (k,dim)
    @param cov: an array containing the covariance matrices (k,dim,dim)
    @return: an array containing the density of probability of a normal law centered
    
    """
    nb_points = len(points)
    k = len(means)
    dim = len(points[0])
    
    #Duplication in order to create matrices k*nb_points*dim
    points_duplicated = np.tile(points, (k,1,1))
    means_duplicated = np.transpose(np.tile(means,(nb_points,1,1)), (1,0,2))
    
    points_centered = points_duplicated - means_duplicated
    if full_covariance:
        cov_inv = np.linalg.inv(cov)
    else:   
        cov_inv = np.tile(np.reciprocal(cov), (dim,dim,1)).T
        identities = np.tile(np.eye(dim,dim), (k,1,1))
        cov_inv = identities * cov_inv / dim
        
    product = np.dot(points_centered,cov_inv)
    product = np.diagonal(np.transpose(product, (0,2,3,1))).T
                         
    points_centered = np.transpose(points_centered, (0,2,1))
    product = np.dot(product,points_centered)  
    product = np.diagonal(np.transpose(product, (0,2,3,1)))
    product = np.diagonal(product).T
    
                         
    det_covariance = np.linalg.det(cov)
    det_covariance = np.reciprocal(det_covariance)
    det_covariance = np.tile(det_covariance, (nb_points,1))
    
    return (2*math.pi)**(-float(dim)/2) * det_covariance**0.5 * np.exp(-0.5 * product)

def full_covariance_matrix_f(points,means,assignements):
    dim = len(points[0])
    nb_points = len(points)
    k = len(means)
    
    #Duplication in order to create matrices k*nb_points*dim
    assignements_duplicated = np.tile(assignements, (dim,1,1)).T
    points_duplicated = np.tile(points, (k,1,1))
    means_duplicated = np.transpose(np.tile(means,(nb_points,1,1)), (1,0,2))
    
    points_centered = points_duplicated - means_duplicated
    points_centered_weighted = np.transpose(assignements_duplicated * points_centered, (0,2,1))
    covariance = np.dot(points_centered_weighted,points_centered)
    covariance = np.transpose(covariance, (0,2,3,1))
    covariance = np.diagonal(covariance).T                     
    
    #Duplication in order to create matrices k*dim*dim
    sum_assignement = np.sum(assignements,axis=0)
    sum_assignement = np.tile(sum_assignement, (dim,dim,1)).T
    sum_assignement = np.reciprocal(sum_assignement)
    
    return covariance*sum_assignement
    
def spherical_covariance_matrix_f(points,means,assignements):
    nb_points = len(points)
    k = len(means)
    dim = len(points[0])
    
    #Duplication in order to create matrices k*nb_points*dim
    assignements_duplicated = np.tile(assignements, (dim,1,1)).T
    points_duplicated = np.tile(points, (k,1,1))
    means_duplicated = np.transpose(np.tile(means,(nb_points,1,1)), (1,0,2))
    
    points_centered = points_duplicated - means_duplicated
    points_centered_weighted = np.transpose(assignements_duplicated * points_centered, (0,2,1))
    covariance = np.dot(points_centered,points_centered_weighted)
    covariance = np.transpose(covariance, (0,2,3,1))
    covariance = np.diagonal(covariance)
    covariance = np.diagonal(covariance).T
    covariance = np.sum(covariance, axis=0)
    
    sum_assignement = np.sum(assignements,axis=0)
    sum_assignement = np.reciprocal(sum_assignement)
    
    return covariance * sum_assignement / dim

def step_E(points,means,cov,prior_prob,full_covariance):
    """
    This method returns the list of the soft assignements of each point to each cluster
    
    @param points: an array of points (nb_points,dim)
    @param means: an array of k points which are the means of the clusters (k,dim)
    @param cov: an array of k arrays which are the covariance matrices (k,dim,dim)
    @param prior_prob: an array of k numbers which are the prior probabilities (k,)
    @return: an array containing the soft assignements of every point (nb_points,k)
    """
    
    k=len(means)
    nb_points = len(points)
    
    normal_matrix = normal_matrix_f(points,means,cov,full_covariance)
    prior_sum = np.dot(normal_matrix,np.transpose(prior_prob))
    prior_sum_reciproc = np.reciprocal(np.tile(prior_sum, (k,1)))
    prior_duplicated = np.tile(prior_prob, (nb_points,1))
    return normal_matrix*prior_duplicated*prior_sum_reciproc.T
  
def step_M(points,assignements,full_covariance):
    """
    This method computes the new position of each mean and each covariance matrix
    
    @param points: an array of points (nb_points,dim)
    @param means: an array of k points which are the means of the clusters (k,dim)
    """
    nb_points = float(len(points))
    dim = len(points[0])
    
    #Phase 1:
    result_inter = np.dot(assignements.T,points)
    sum_assignements = np.sum(assignements,axis=0)
    sum_assignements_final = np.reciprocal(np.tile(sum_assignements, (dim,1)).T)
    
    means = result_inter*sum_assignements_final
    
    #Phase 2:
    if full_covariance:
        cov = full_covariance_matrix_f(points,means,assignements)
#        list_cov = np.asarray([covariance_matrix(list_points,list_means[i],np.transpose(list_soft_assignement)[i]) for i in range(k)])
    else:
        cov = spherical_covariance_matrix_f(points,means,assignements)
        print(cov)
                    
    #Phase 3:
    list_prior_prob = 1/nb_points * np.sum(assignements, axis=0)
    
    return means,cov,list_prior_prob
    
def create_graph(points,means,cov,assignements,full_covariance,t):
    """
    This method draws a 2D graph displaying the clusters and their means and saves it as a PNG file.
    If points have more than two coordinates, then it will be a projection including only the first coordinates.
    
    @param points: an array of points
    @param means: an array of k points which are the means of the clusters
    @param k: the figure number
    """
    
    k=len(means)
    nb_points=len(points)
    dim=len(points[0])
    
    x_points = [[] for i in range(k)]
    y_points = [[] for i in range(k)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111,aspect='equal')
    
    for i in range(k):
        if full_covariance:
            covariance = cov[i]
        else:
            covariance = cov[i]*np.eye(dim,dim)
        lambda_, v = np.linalg.eig(covariance)
        lambda_ = np.sqrt(lambda_)
        ell = Ellipse(xy=(means[i][0], means[i][1]),
                  width=lambda_[0]*2, height=lambda_[1]*2,
                  angle=np.rad2deg(np.arccos(v[0, 0])))
        ell.set_facecolor('none')
        ell.set_edgecolor('k')
        ax.add_artist(ell)
        x_points[i] = [points[j][0] for j in range(nb_points) if (assignements[j][i]>0.7)]        
        y_points[i] = [points[j][1] for j in range(nb_points) if (assignements[j][i]>0.7)]

        ax.plot(x_points[i],y_points[i],'o',alpha = 0.2)
        ax.plot(means[i][0],means[i][1],'x')
        
        
    
    titre = 'figure_' + str(t)
    plt.savefig(titre)
    plt.close("all")

def log_likelihood(points,means,cov,prior_prob,full_covariance):
    """
    This method returns the log likelihood at the end of the k_means.
    
    @param list_points: an array of points
    @param list_means: an array of k points which are the means of the clusters
    @param list_cov: an array of k arrays of size NxN which are the covariance matrices
    @param list_prior_prob: an array of k numbers which are the prior probabilities
    @return: log likelihood measurement (float)
    """
    normal_matrix = normal_matrix_f(points,means,cov,full_covariance)
    return np.sum(np.log10(np.dot(normal_matrix,np.transpose(prior_prob))))
    
    
def GMM(points,k,draw_graphs=False,initialization=CSVreader.initialization_plus_plus,epsilon=0.00001,full_covariance=True):
    """
    This method returns an array of k points which will be used in order to
    initialize a k_means algorithm
    
    @param list_points: an array of points
    @param k: the number of clusters
    @param draw_graphs: a boolean to allow the algorithm to draw graphs if True
    @param initialization: a method in order to initialize the k_means algorithm
    @return: an array of k points which are means of clusters
    """
    
    #K-means++ initialization
    means = initialization(points,k)
    points = points[:,0:-1]
    means = means[:,0:-1]
    
    dim = len(points[0])
    
    if full_covariance:
        cov = np.tile(np.eye(dim,dim), (k,1,1))
    else:
        cov = np.ones(k)
    
    prior_prob = np.ones(k) * 1/float(k)
    
    log_like = log_likelihood(points,means,cov,prior_prob,full_covariance)
                             
    resume_iter = True  
    t=0       
    
    #K-means beginning
    while resume_iter:
                
        log_like_pre = log_like
        assignements = step_E(points,means,cov,prior_prob,full_covariance)
        means,cov,prior_prob = step_M(points,assignements,full_covariance)
        log_like = log_likelihood(points,means,cov,prior_prob,full_covariance)
        
        #Graphic part
        if draw_graphs:
            create_graph(points,means,cov,assignements,full_covariance,t)
        
        t+=1
        
        resume_iter = abs((log_like - log_like_pre)/log_like) > epsilon
    print(t)
    
if __name__ == '__main__':
    
    #Lecture du fichier
    points = CSVreader.read("D:/Mines/Cours/Stages/Stage_ENS/Problem/EMGaussienne.data")
    
    #k-means
    k=5
    GMM(points,k,draw_graphs=False,full_covariance=True)