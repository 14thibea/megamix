# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 12:08:35 2017

@author: Calixi
"""

import CSVreader
import matplotlib.pyplot as plt
import numpy as np
import random

#A distance to not take in account the last index (which is the cluster number)
def dist(array1,array2):
    point1 = array1[:-1]
    point2 = array2[:-1]
    return np.linalg.norm(point1-point2)

def initialization_plus_plus(list_points,k):
    
    nb_points = len(list_points)
    v = np.arange(nb_points)/nb_points
    list_points[:,-1] = v
               
    list_means = [[] for i in range(k)]
    
    for i in range(k): 
        total_dst = 0          
        dst_square_total = 0
        #Choice of a new seed
        value_chosen = random.uniform(0,1)
        idx_point = 0
        value = 0
        while (value<value_chosen) and (idx_point+1<nb_points):
            idx_point +=1
            value = list_points[idx_point][-1]
        list_means[i] = list_points[idx_point]
        
        #Calculation of distances for each point in order to find the probabilities to choose each point
        for point in list_points:
            minimum = dist(list_means[0],point)
            for mean in list_means:
                if (len(mean)!=0):
                    dst = dist(mean,point)
                    if (dst<minimum):
                        minimum = dst
            total_dst += dst**2
            point[-1] = total_dst
            dst_square_total += dst**2
        
        list_points[:,-1] = list_points[:,-1]/dst_square_total
                   
    return list_means

#Cluster attribution
def step_E(list_points,list_means):
    point_number = 0
    for point in list_points:
        mean_number = 0
        cluster_number = 0
        minimum = dist(list_means[0],point)
        for mean in list_means:
            dst = dist(mean,point)
            if (dst<minimum):
                minimum = dst
                cluster_number = mean_number
            mean_number+=1
        list_points[point_number][-1] = cluster_number
        point_number+=1
        

#Mean positions calculation        
def step_M(list_points,list_means):
    k = len(list_means)
    length = len(list_points)
    for j in range(k):
        list_set = [list_points[i] for i in range(length) if (list_points[i][-1]==j)]
        list_means[j] = np.mean(list_set, axis=0)
        
#Only for 2D vectors
def create_graph(k,list_points,list_means,t):
    x_points = [[] for i in range(k)]
    y_points = [[] for i in range(k)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for i in range(len(list_means)):
        for point in list_points:
            if (point[-1]==i):
                x_points[i].append(point[0])
                y_points[i].append(point[1])

        ax.plot(x_points[i],y_points[i],'o')
        ax.plot(list_means[i][0],list_means[i][1],'x')
    
    titre = 'figure_' + str(t)
    plt.savefig(titre)

def k_means(list_points,k):
    #initialization
#    
#    nb_points = len(list_points)
#    idx = np.random.randint(nb_points,size = k)
#    list_means = list_points[idx,:]
#    list_means_pre = list_means.copy()
#    
#    resume_iter = True
    
    #K-means++ initialization
               
    list_means = initialization_plus_plus(list_points,k)
    list_means_pre = list_means.copy()
    
    resume_iter = True  
    t=0       
    
    #K-means beginning
    
    while resume_iter:
                        
        step_E(list_points,list_means)        
        list_means_pre = list_means.copy()
        step_M(list_points,list_means)
        
    #Graphic part            
        create_graph(k,list_points,list_means,t)
        
        t+=1        
        resume_iter = not np.array_equal(list_means,list_means_pre)

#Lecture du fichier

list_points = CSVreader.read("D:/Mines/Cours/Stages/Stage_ENS/Problem/EMGaussienne.data")
#k-means

k=3
k_means(list_points,k)

    
