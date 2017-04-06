# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 12:08:35 2017

@author: Calixi
"""

import Initializations as Init
import matplotlib.pyplot as plt
import numpy as np
import CSVreader


def dist_matrix(list_points,list_means,k):
    """
    This method computes all the distances between the points and the actual means
    of the clusters, taking in account that the last coordinate is the cluster number
    
    @param list_points: an array of points
    @param list_means: an array of k points which are the means of the clusters
    @return: a matrix which contains the distances between the ith point and the jth center
    """
    
    nb_points = len(list_points)
    
    real_points = list_points[:,0:-1]
    square_points_distances = np.sum(real_points*real_points, axis=1)
    A = np.transpose(np.asmatrix([square_points_distances for i in range(k)]))
    # A is a nb_points * k matrix. Aij is the squared norm of the ith point of list_points
    
    real_means = list_means[:,0:-1]
    real_means = real_means[0:k:1]
    real_means = np.asarray(real_means)
    square_means_distances = np.sum(real_means*real_means, axis=1)
    B = np.asmatrix([square_means_distances for i in range(nb_points)])
    # B is a nb_points * k matrix. Bij is the squared norm of the jth point of list_means   
    
    C = np.dot(real_points,np.transpose(real_means))
    # C is a nb_points * k matrix. Cij is the scalar product of the ith point of list_points and the jth point of list_means
    
    return np.sqrt(A+B-2*C)

def step_E(points,means):
    """
    This method assign a cluster number to each point by changing its last coordinate
    Ex : if P belongs to the first cluster, then P[-1] = 0.
    
    @param points: an array of points
    @param means: an array of k points which are the means of the clusters
    """
    k = len(means)
    nb_points = len(points)
    M = dist_matrix(points,means,k)
    for i in range(nb_points):
        index_min = np.argmin(M[i]) #the cluster number of the ith point is index_min
        if (isinstance(index_min,np.int64)):
            points[i][-1] = index_min
        else: #Happens when two points are equally distant from a cluster mean
            points[i][-1] = index_min[0]
  
def step_M(points,means):
    """
    This method computes the new position of each means by minimizing the distortion
    
    @param points: an array of points
    @param means: an array of k points which are the means of the clusters
    """
    k = len(means)
    nb_points = len(points)
    for j in range(k):
        sets = [points[i] for i in range(nb_points) if (points[i][-1]==j)]
        means[j] = np.mean(sets, axis=0)
        
def create_graph(points,means,t):
    """
    This method draws a 2D graph displaying the clusters and their means and saves it as a PNG file.
    If points have more than two coordinates, then it will be a projection including only the first coordinates.
    
    @param points: an array of points
    @param means: an array of k points which are the means of the clusters
    @param t: the figure number
    """
    
    k=len(means)
    
    x_points = [[] for i in range(k)]
    y_points = [[] for i in range(k)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for i in range(k):
        x_points[i] = [point[0] for point in points if (point[-1]==i)]        
        y_points[i] = [point[1] for point in points if (point[-1]==i)]

        ax.plot(x_points[i],y_points[i],'o')
        ax.plot(means[i][0],means[i][1],'x')
    
    titre = 'figure_' + str(t)
    plt.savefig(titre)
    plt.close("all")

def distortion(points,means):
    """
    This method returns the distortion measurement at the end of the k_means.
    
    @param points: an array of points
    @param means: an array of k points which are the means of the clusters
    @return: distortion measurement (float)
    """
    k=len(means)
    distortion = 0
    for i in range(k):
        sets = [point for point in points if (point[-1]==i)]
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
                        
        step_E(points,means)        
        means_pre = means.copy()
        step_M(points,means)
        
        #Graphic part
        if draw_graphs:
            create_graph(points,means,t)
        
        t+=1        
        resume_iter = not np.array_equal(means,means_pre)
        
    return(means)

if __name__ == '__main__':
    
    #Lecture du fichier
    points = CSVreader.read("D:/Mines/Cours/Stages/Stage_ENS/Problem/EMGaussienne.test")
    
    #k-means
    k=3
    means = k_means(points,k,draw_graphs=True,initialization="plus")
    print(means)