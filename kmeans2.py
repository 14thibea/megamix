# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 12:08:35 2017

@author: Calixi
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def dist(array1,array2):
    point1 = array1[:-1]
    point2 = array2[:-1]
    return np.linalg.norm(point1-point2)
    
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
        list_points[point_number][dimension] = cluster_number
        point_number+=1
        
        
def step_M(list_points,list_means):
    k = len(list_means)
    length = len(list_points)
    for j in range(k):
        list_set = [list_points[i] for i in range(length) if (list_points[i][dimension]==j)]
        list_means[j] = np.mean(list_set, axis=0)
        
#Only for two dimension vectors
def create_graph(list_points,list_means,t):
    x_points = [[] for i in range(k)]
    y_points = [[] for i in range(k)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for i in range(len(list_means)):
        for point in list_points:
            if (point[dimension]==i):
                x_points[i].append(point[0])
                y_points[i].append(point[1])

        ax.plot(x_points[i],y_points[i],'o')
        ax.plot(list_means[i][0],list_means[i][1],'x')
    
    titre = 'figure_' + str(t)
    plt.savefig(titre)

def k_means(list_points,k):
    #initialisation

    idx = np.random.randint(nb_points,size = k)
    list_means = list_points[idx,:]
    list_means_pre = list_means.copy()
    
    resume_iter = True 
       
    #K-means beginning
    
    t=0
    while resume_iter:
                        
        step_E(list_points,list_means)        
        list_means_pre = list_means.copy()
        step_M(list_points,list_means)
        
    #Graphic part            
        create_graph(list_points,list_means,t)
        
        t+=1
        
        resume_iter = not np.array_equal(list_means,list_means_pre)

#Lecture du fichier

fichier = pd.read_csv("D:/Mines/Cours/Stages/Stage_ENS/Problem/EMGaussienne.data",sep = " ")
matrix = fichier.as_matrix()
nb_points = len(matrix)
dimension = len(matrix[0])

list_points = np.zeros((nb_points,dimension+1))
list_points[:,:-1] = matrix

#k-means

k=5
k_means(list_points,k)

    
