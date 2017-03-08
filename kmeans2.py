# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 12:08:35 2017

@author: Calixi
"""

import CSVreader
import matplotlib.pyplot as plt
import numpy as np
import random

def dist_matrix(list_points,list_means,k):
    """
    This method computes all the distances between the points and the actual means
    of the clusters, taking in account that the last coordinate is the cluster number
    
    @param list_points: an array of points
    @param list_means: an array of k points which are the means of the clusters
    @return: a matrix which contains the distances between the ith point and the jth center
    """
    
    nb_points = len(list_points)
    
    real_points = list_points[:,0:2]
    square_points_distances = np.sum(real_points*real_points, axis=1)
    A = np.transpose(np.asmatrix([square_points_distances for i in range(k)]))
    # A is a nb_points * k matrix. Aij is the squared norm of the ith point of list_points
    
    real_means = list_means[:,0:2]
    real_means = real_means[0:k:1]
    real_means = np.asarray(real_means)
    square_means_distances = np.sum(real_means*real_means, axis=1)
    B = np.asmatrix([square_means_distances for i in range(nb_points)])
    # B is a nb_points * k matrix. Bij is the squared norm of the jth point of list_means   
    
    C = np.dot(real_points,np.transpose(real_means))
    # C is a nb_points * k matrix. Cij is the scalar product of the ith point of list_points and the jth point of list_means
    
    return np.sqrt(A+B-2*C)

def initialization_random(list_point,k):
    """
    This method returns an array of k points which will be used in order to
    initialize a k_means algorithm
    
    @param list_points: an array of points
    @param k: the number of clusters
    @return: an array of k points which are means of clusters
    """

    nb_points = len(list_points)
    idx = np.random.randint(nb_points,size = k)
    return list_points[idx,:]

def initialization_plus_plus(list_points,k):
    """
    This method returns an array of k points which will be used in order to
    initialize a k_means algorithm
    
    @param list_points: an array of points
    @param k: the number of clusters
    @return: an array of k points which are means of clusters
    """
    nb_points = len(list_points)
    dimension = len(list_points[0])
    probability_vector = np.arange(nb_points)/nb_points #All points have the same probability to be chosen the first time
    list_points[:,-1] = probability_vector
               
    list_means = np.zeros((k,dimension))
    
    for i in range(k): 
        total_dst = 0          
        
        #Choice of a new seed
        value_chosen = random.uniform(0,1)
        idx_point = 0
        value = 0
        while (value<value_chosen) and (idx_point+1<nb_points):
            idx_point +=1
            value = list_points[idx_point][-1]
        list_means[i] = list_points[idx_point]
        
        #Calculation of distances for each point in order to find the probabilities to choose each point
        M = dist_matrix(list_points,list_means,i+1)
        
        for point in list_points:
            dst = np.min(M[i])
            total_dst += dst**2
            point[-1] = total_dst
        
        list_points[:,-1] = list_points[:,-1]/total_dst

    return list_means

def step_E(list_points,list_means):
    """
    This method assign a cluster number to each point by changing its last coordinate
    Ex : if P belongs to the first cluster, then P[-1] = 0.
    
    @param list_points: an array of points
    @param list_means: an array of k points which are the means of the clusters
    """
    M = dist_matrix(list_points,list_means,k)
    nb_points = len(list_points)
    for i in range(nb_points):
        index_min = np.argmin(M[i]) #the cluster number of the ith point is index_min
        if (isinstance(index_min,np.int64)):
            list_points[i][-1] = index_min
        else:
            list_points[i][-1] = index_min[0]
  
def step_M(list_points,list_means):
    """
    This method computes the new position of each means by minimizing the distortion
    
    @param list_points: an array of points
    @param list_means: an array of k points which are the means of the clusters
    """
    k = len(list_means)
    length = len(list_points)
    for j in range(k):
        list_set = [list_points[i] for i in range(length) if (list_points[i][-1]==j)]
        list_means[j] = np.mean(list_set, axis=0)
        
def create_graph(list_points,list_means,t):
    """
    This method draws a 2D graph displaying the clusters and their means and saves it as a PNG file.
    If points have more than two coordinates, then it will be a projection including only the first coordinates.
    
    @param list_points: an array of points
    @param list_means: an array of k points which are the means of the clusters
    @param k: the figure number
    """
    
    k=len(list_means)
    
    x_points = [[] for i in range(k)]
    y_points = [[] for i in range(k)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for i in range(k):
        x_points[i] = [point[0] for point in list_points if (point[-1]==i)]        
        y_points[i] = [point[1] for point in list_points if (point[-1]==i)]

        ax.plot(x_points[i],y_points[i],'o')
        ax.plot(list_means[i][0],list_means[i][1],'x')
    
    titre = 'figure_' + str(t)
    plt.savefig(titre)
    plt.close("all")

def distortion(list_points,list_means):
    """
    This method returns the distortion measurement at the end of the k_means.
    
    @param list_points: an array of points
    @param list_means: an array of k points which are the means of the clusters
    @return: distortion measurement (float)
    """
    distortion = 0
    for i in range(k):
        list_set = [point for point in list_points if (point[-1]==i)]
        list_set = np.asarray(list_set)
        M = dist_matrix(list_set,np.asmatrix(list_means[i]),1)
        distortion += np.sum(M)
    return distortion
    
    
def k_means(list_points,k,draw_graphs=False,initialization=initialization_plus_plus):
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
    list_means_pre = list_means.copy()
    
    resume_iter = True  
    t=0       
    
    #K-means beginning
    while resume_iter:
                        
        step_E(list_points,list_means)        
        list_means_pre = list_means.copy()
        step_M(list_points,list_means)
        
        #Graphic part
        if draw_graphs:
            create_graph(list_points,list_means,t)
        
        t+=1        
        resume_iter = not np.array_equal(list_means,list_means_pre)

    print(distortion(list_points,list_means))

if __name__ == '__main__':
    
    #Lecture du fichier
    list_points = CSVreader.read("D:/Mines/Cours/Stages/Stage_ENS/Problem/EMGaussienne.data")
    
    #k-means
    k=3
    k_means(list_points,k,draw_graphs=False)