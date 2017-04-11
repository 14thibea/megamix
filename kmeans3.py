# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 14:20:27 2017

@author: Elina Thibeau-Sutre
"""

import Initializations as Init
import utils

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics.pairwise import euclidean_distances
import pickle
#from scipy.misc import logsumexp


def dist_matrix(points,means,k):
    """
    This method computes all the distances between the points and the actual means
    of the clusters, taking in account that the last coordinate is the cluster number
    
    @param points: an array (n_points,dim)
    @param means: an array containing the means of the clusters (n_components,dim)
    @return: a matrix which contains the distances between the ith point and the jth center
    (n_points,n_components)
    """

    real_means = means[:k:]
    dist_matrix = euclidean_distances(points,real_means)
    
    return dist_matrix

def step_E(points,means):
    """
    This method assign a cluster number to each point by changing its last coordinate
    Ex : if P belongs to the first cluster, then P[-1] = 0.
    
    @param points: an array (n_points,dim)
    @param means: an array containing the means of the clusters (n_components,dim)
    """
    k = len(means)
    n_points = len(points)
    assignements = np.zeros((n_points,k))
    
    M = dist_matrix(points,means,k)
    for i in range(n_points):
        index_min = np.argmin(M[i]) #the cluster number of the ith point is index_min
        if (isinstance(index_min,np.int64)):
            assignements[i][index_min] = 1
        else: #Happens when two points are equally distant from a cluster mean
            assignements[i][index_min[0]] = 1
            
    return assignements
  
def step_M(points,means,assignements):
    """
    This method computes the new position of each means by minimizing the distortion
    
    @param points: an array (n_points,dim)
    @param means: an array containing the means of the clusters (n_components,dim)
    """
    k = len(means)
    n_points,dim = points.shape
    
    for i in range(k):
        sets = assignements[:,i:i+1]
        n_sets = np.sum(sets)
        sets = points * np.tile(sets, (1,dim))
        means[i] = np.mean(sets, axis=0)*n_points/n_sets
        
    return means
        
def create_graph(points,means,assignements,t):
    """
    This method draws a 2D graph displaying the clusters and their means and saves it as a PNG file.
    If points have more than two coordinates, then it will be a projection including only the first coordinates.
    
    @param points: an array (n_points,dim)
    @param means: an array containing the means of the clusters (n_components,dim)
    @param t: the figure number
    """
    
    k=len(means)
    n_points = len(points)
    
    x_points = [[] for i in range(k)]
    y_points = [[] for i in range(k)]
    
    dist = distortion(points,means,assignements)
    
    fig = plt.figure()
    plt.title("distortion = " + str(dist))
    ax = fig.add_subplot(111)
    
    for i in range(k):
        x_points[i] = [points[j][0] for j in range(n_points) if (assignements[j][i] == 1)]        
        y_points[i] = [points[j][1] for j in range(n_points) if (assignements[j][i] == 1)]

        ax.plot(x_points[i],y_points[i],'o')
        ax.plot(means[i][0],means[i][1],'kx')
        
        
    dir_path = 'k_means/'
    directory = os.path.dirname(dir_path)

    try:
        os.stat(directory)
    except:
        os.mkdir(directory)  
    
    titre = directory + '/figure_' + str(t)

    plt.savefig(titre)
    plt.close("all")

def distortion(points,means,assignements):
    """
    This method returns the distortion measurement at the end of the k_means.
    
    @param points: an array (n_points,dim)
    @param means: an array containing the means of the clusters (n_components,dim)
    @return: distortion measurement (float)
    """
    k=len(means)
    n_points=len(points)
    distortion = 0
    for i in range(k):
        sets = [points[j] for j in range(n_points) if (assignements[j][i]==1)]
        sets = np.asarray(sets)
        M = dist_matrix(sets,np.asmatrix(means[i]),1)
        distortion += np.sum(M)
    return distortion
    
    
def k_means(points,k,draw_graphs=False,initialization = "plus"):
    """
    This method returns an array of k points which will be used in order to
    initialize a k_means algorithm
    
    @param points: an array (n_points,dim)
    @param k: the number of clusters
    @param draw_graphs: a boolean to allow the algorithm to draw graphs if True
    @param initialization: a string in ['random','plus']
    @return: the means of the clusters (n_components,dim)
    """
    
    #K-means++ initialization
    if (initialization == "random"):
        means = Init.initialization_random(points,k)
    elif (initialization == "plus"):
        means = Init.initialization_plus_plus(points,k)
    else:
        raise ValueError("Invalid value for 'initialization': %s "
                             "'initialization' should be in "
                             "['random', 'plus']"
                              % initialization)
    
    means_pre = means.copy()
    
    resume_iter = True  
    t=0       
    
    #K-means beginning
    while resume_iter:
                      
        assignements = step_E(points,means)        
        means_pre = means.copy()
        means = step_M(points,means,assignements)
        
        #Graphic part
        if draw_graphs:
            create_graph(points,means,assignements,t)
        
        t+=1        
        resume_iter = not np.array_equal(means,means_pre)
        
    return means,assignements

if __name__ == '__main__':
    
    #Lecture du fichier
    points = utils.read("D:/Mines/Cours/Stages/Stage_ENS/Code/data/EMGaussienne.data")
    
#    path = 'D:/Mines/Cours/Stages/Stage_ENS/Code/data/data.pickle'
#    with open(path, 'rb') as fh:
#        data = pickle.load(fh)
#    
#    N=5000
#        
#    points = data['BUC']
#    points = points[:N:]
#    points = points[:,0:2]
#    
    #k-means
    k=4
    
    for i in range(20):
        means,assignements = k_means(points,k,draw_graphs=False,initialization="plus")
        create_graph(points,means,assignements,i)