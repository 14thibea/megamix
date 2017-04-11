# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 12:08:35 2017

@author: Calixi
"""

import Initializations as Init
import utils

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
#from scipy.misc import logsumexp


def dist_matrix(points,means,k):
    """
    This method computes all the distances between the points and the actual means
    of the clusters, taking in account that the last coordinate is the cluster number
    
    @param list_points: an array of points
    @param list_means: an array of k points which are the means of the clusters
    @return: a matrix which contains the distances between the ith point and the jth center
    """
    
    n_points = len(points)
    
    square_points_distances = np.sum(points*points, axis=1)
    A = np.tile(square_points_distances, (k,1)).T
    # A is a nb_points * k matrix. Aij is the squared norm of the ith point of list_points
    
    real_means = means[0:k:1]
    real_means = np.asarray(real_means)
    square_means_distances = np.sum(real_means*real_means, axis=1)
    B = np.tile(square_means_distances,(n_points,1))
    # B is a nb_points * k matrix. Bij is the squared norm of the jth point of list_means   
    
    C = np.dot(points,real_means.T)
    # C is a nb_points * k matrix. Cij is the scalar product of the ith point of list_points and the jth point of list_means
    
    D = A+B-2*C
    
    return np.sqrt(D)

def step_E(points,means):
    """
    This method assign a cluster number to each point by changing its last coordinate
    Ex : if P belongs to the first cluster, then P[-1] = 0.
    
    @param points: an array of points
    @param means: an array of k points which are the means of the clusters
    """
    k = len(means)
    n_points = len(points)
    assignements = np.zeros(n_points)
    
    M = dist_matrix(points,means,k)
    for i in range(n_points):
        index_min = np.argmin(M[i]) #the cluster number of the ith point is index_min
        if (isinstance(index_min,np.int64)):
            assignements[i] = index_min
        else: #Happens when two points are equally distant from a cluster mean
            assignements[i] = index_min[0]
            
    return assignements
  
def step_M(points,means,assignements):
    """
    This method computes the new position of each means by minimizing the distortion
    
    @param points: an array of points
    @param means: an array of k points which are the means of the clusters
    """
    k = len(means)
    n_points = len(points)
    
    for j in range(k):
        sets = [points[i] for i in range(n_points) if (assignements[i]==j)]
        means[j] = np.mean(sets, axis=0)
        
def create_graph(points,means,assignements,t):
    """
    This method draws a 2D graph displaying the clusters and their means and saves it as a PNG file.
    If points have more than two coordinates, then it will be a projection including only the first coordinates.
    
    @param points: an array of points
    @param means: an array of k points which are the means of the clusters
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
        x_points[i] = [points[j][0] for j in range(n_points) if (assignements[j]==i)]        
        y_points[i] = [points[j][1] for j in range(n_points) if (assignements[j]==i)]

        ax.plot(x_points[i],y_points[i],'o')
        ax.plot(means[i][0],means[i][1],'x')
        
        
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
    
    @param points: an array of points
    @param means: an array of k points which are the means of the clusters
    @return: distortion measurement (float)
    """
    k=len(means)
    n_points=len(points)
    distortion = 0
    for i in range(k):
        sets = [points[j] for j in range(n_points) if (assignements[j]==i)]
        sets = np.asarray(sets)
        M = dist_matrix(sets,np.asmatrix(means[i]),1)
        distortion += np.sum(M)
    return distortion
    
    
def k_means(points,k,draw_graphs=False,initialization = "plus"):
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
    if (initialization == "rand"):
        means = Init.initialization_random(points,k)
    elif (initialization == "plus"):
        means = Init.initialization_plus_plus(points,k)
    else:
        raise ValueError("Invalid value for 'initialization': %s "
                             "'initialization' should be in "
                             "['rand', 'plus']"
                              % initialization)
    
    means_pre = means.copy()
    
    resume_iter = True  
    t=0       
    
    #K-means beginning
    while resume_iter:
                      
        assignements = step_E(points,means)        
        means_pre = means.copy()
        step_M(points,means,assignements)
        
        #Graphic part
        if draw_graphs:
            create_graph(points,means,assignements,t)
        
        t+=1        
        resume_iter = not np.array_equal(means,means_pre)
        
    return means,assignements

if __name__ == '__main__':
    
    #Lecture du fichier
    points = utils.read("D:/Mines/Cours/Stages/Stage_ENS/Code/Problem/data/EMGaussienne.data")
   
    path = 'D:/Mines/Cours/Stages/Stage_ENS/Code/Problem/data/data.pickle'
    with open(path, 'rb') as fh:
        data = pickle.load(fh)
    
    N=5000
        
    points = data['BUC']
    points = points[:N:]
    points = points[:,0:2]
    #k-means
    k=4
    
    k_means(points,k)

#    for i in range(20):
#        means,assignements = k_means(points,k,draw_graphs=False,initialization="plus")
#        create_graph(points,means,assignements,i)