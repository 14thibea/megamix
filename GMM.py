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

def density_norm(point,mean,cov):
    """
    This method computes the density of probability of a normal law centered on mean
    and with covariance point
    
    @param point: a nD array
    @param mean: a nD array
    @param cov: a nxn array, the covariance matrix
    @return: the density of probability
    """
    dim = float(len(point))
    diff = point-mean
    det = np.linalg.det(cov)
    scalar = (2*math.pi)**(dim/2) * math.sqrt(det)
    return 1.0/scalar * math.exp(-0.5*np.dot(np.dot(np.transpose(diff),np.linalg.inv(cov)),diff))

def covariance_matrix(list_points,mean,soft_assignements):
    dim = len(list_points[0])
    nb_points = len(list_points)
    mean = np.asarray([mean for i in range(nb_points)])
    points_centered = list_points - mean
    sum_soft_assignements = np.sum(soft_assignements)
    soft_assignements_rep = np.transpose(np.asarray([soft_assignements for i in range(dim)]))
    return 1/sum_soft_assignements * np.dot(np.transpose(soft_assignements_rep*points_centered),points_centered)

def step_E(list_points,list_means,list_cov,list_prior_prob):
    """
    This method returns the list of the soft assignements of each point to each cluster
    
    @param list_points: an array of points (size N)
    @param list_means: an array of k points which are the means of the clusters
    @param list_cov: an array of k arrays of size NxN which are the covariance matrices
    @param list_prior_prob: an array of k numbers which are the prior probabilities
    @return: an array of size Nxk containing the soft assignements of every point 
    """
    
    k=len(list_means)
    nb_points = len(list_points)
    normal_matrix = np.asarray([[density_norm(point,list_means[i],list_cov[i]) for i in range(k)] for point in list_points])
    prior_sum = np.dot(normal_matrix,np.transpose(list_prior_prob))
    prior_sum_reciproc = np.transpose(np.reciprocal(np.asarray([prior_sum for i in range(k)])))
    prior_duplicated = np.asarray([list_prior_prob for i in range(nb_points)])
    return normal_matrix*prior_duplicated*prior_sum_reciproc
  
def step_M(list_points,list_soft_assignement,full_covariance):
    """
    This method computes the new position of each mean and each covariance matrix
    
    @param list_points: an array of points
    @param list_means: an array of k points which are the means of the clusters
    """
    k = len(list_soft_assignement[0])
    nb_points = float(len(list_points))
    dim = len(list_points[0])
    
    #Phase 1:
    result_inter = np.dot(np.transpose(list_soft_assignement),list_points)
    sum_assignements = np.sum(list_soft_assignement,axis=0)
    sum_assignements_final = np.reciprocal(np.transpose(np.asarray([sum_assignements for i in range(dim)])))
    
    list_means = result_inter*sum_assignements_final
    
    #Phase 2:
    if full_covariance:
        list_cov = np.asarray([covariance_matrix(list_points,list_means[i],np.transpose(list_soft_assignement)[i]) for i in range(k)])
    else:
        list_cov = np.asarray([np.sum(covariance_matrix(list_points,list_means[i],np.transpose(list_soft_assignement)[i])) *1/k * np.eye(dim,dim) for i in range(k)])
    
    #Phase 3:
    list_prior_prob = 1/nb_points * np.sum(list_soft_assignement, axis=0)
    
    return list_means,list_cov,list_prior_prob
    
def create_graph(list_points,list_means,list_cov,list_soft_assignements,t):
    """
    This method draws a 2D graph displaying the clusters and their means and saves it as a PNG file.
    If points have more than two coordinates, then it will be a projection including only the first coordinates.
    
    @param list_points: an array of points
    @param list_means: an array of k points which are the means of the clusters
    @param k: the figure number
    """
    
    k=len(list_means)
    nb_points=len(list_points)
    
    x_points = [[] for i in range(k)]
    y_points = [[] for i in range(k)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111,aspect='equal')
    
    for i in range(k):
        cov = list_cov[i]
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        ell = Ellipse(xy=(list_means[i][0], list_means[i][1]),
                  width=lambda_[0]*2, height=lambda_[1]*2,
                  angle=np.rad2deg(np.arccos(v[0, 0])))
        ell.set_facecolor('none')
        ell.set_edgecolor('k')
        ax.add_artist(ell)
        x_points[i] = [list_points[j][0] for j in range(nb_points) if (list_soft_assignements[j][i]>0.7)]        
        y_points[i] = [list_points[j][1] for j in range(nb_points) if (list_soft_assignements[j][i]>0.7)]

        ax.plot(x_points[i],y_points[i],'o',alpha = 0.2)
        ax.plot(list_means[i][0],list_means[i][1],'x')
        
        
    
    titre = 'figure_' + str(t)
    plt.savefig(titre)
    plt.close("all")

def log_likelihood(list_points,list_means,list_cov,list_prior_prob):
    """
    This method returns the log likelihood at the end of the k_means.
    
    @param list_points: an array of points
    @param list_means: an array of k points which are the means of the clusters
    @param list_cov: an array of k arrays of size NxN which are the covariance matrices
    @param list_prior_prob: an array of k numbers which are the prior probabilities
    @return: log likelihood measurement (float)
    """
    k=len(list_means)
    normal_matrix = np.asarray([[density_norm(point,list_means[i],list_cov[i]) for i in range(k)] for point in list_points])
    return np.sum(np.log10(np.dot(normal_matrix,np.transpose(list_prior_prob))))
    
    
def GMM(list_points,k,draw_graphs=False,initialization=CSVreader.initialization_plus_plus,epsilon=0.00001,full_covariance=True):
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
    list_means = initialization(list_points,k)
    list_points = list_points[:,0:-1]
    list_means = list_means[:,0:-1]
    
    dim = len(list_points[0])
    list_cov = np.asarray([np.eye(dim,dim) for i in range(k)])
    
    list_prior_prob = np.ones(k) * 1/float(k)
    
    log_like = log_likelihood(list_points,list_means,list_cov,list_prior_prob)
                             
    resume_iter = True  
    t=0       
    
    #K-means beginning
    while resume_iter:
                
        log_like_pre = log_like
        list_soft_assignements = step_E(list_points,list_means,list_cov,list_prior_prob)
        list_means,list_cov,list_prior_prob = step_M(list_points,list_soft_assignements,full_covariance)
        log_like = log_likelihood(list_points,list_means,list_cov,list_prior_prob)
        
        #Graphic part
        if draw_graphs:
            create_graph(list_points,list_means,list_cov,list_soft_assignements,t)
        
        t+=1
        
        resume_iter = abs((log_like - log_like_pre)/log_like) > epsilon
        print(t, log_likelihood(list_points,list_means,list_cov,list_prior_prob))

if __name__ == '__main__':
    
    #Lecture du fichier
    list_points = CSVreader.read("D:/Mines/Cours/Stages/Stage_ENS/Problem/EMGaussienne.data")
    
    #k-means
    k=4
    GMM(list_points,k,draw_graphs=True,full_covariance=False)