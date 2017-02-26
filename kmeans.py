# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 17:22:14 2017

@author: Calixi
"""

import csv
import random
import matplotlib.pyplot as plt
from scipy.spatial import distance

    
def step_E(list_points,list_means,list_sets):
    for point in list_points:
        mean_number = 0
        cluster_number = 0
        minimum = distance.euclidean(list_means[0],point)
        for mean in list_means:
            dst = distance.euclidean(mean,point)
            if (dst<minimum):
                minimum = dst
                cluster_number = mean_number
            mean_number+=1
        list_sets[cluster_number].append(point)
        
        
def step_M(k,list_sets,list_means):
    for i in range(k):
        length = len(list_sets[i])       
        x=0
        y=0
        #############
        #CHANGE HERE#
        #############
        for j in range(len(list_sets[i])):
            point = list_sets[i][j]
            x+=point[0]/length
            y+=point[1]/length
        list_means[i] = (x,y)
        

def create_graph(k,list_sets):
    x_points = [[] for i in range(k)]
    y_points = [[] for i in range(k)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for i in range(k):
        for j in range(len(list_sets[i])):
            point = list_sets[i][j]
            x_points[i].append(point[0])
            y_points[i].append(point[1])

        ax.plot(x_points[i],y_points[i],'o')
        ax.plot(list_means[i][0],list_means[i][1],'x')
    
    titre = 'figure_' + str(t)
    plt.savefig(titre)

#Lecture du fichier

fichier = open("D:/Mines/Cours/Stages/Stage_ENS/Problem/EMGaussienne.data",'r')
reader = csv.reader(fichier,delimiter=" ")
list_points = []
first_iter = True
for row in reader:
    if first_iter:
        print(len(row))
    a,b = row
    point = (float(a),float(b))
    list_points.append(point)
    first_iter = False
fichier.close()


#k-means

#initialisation

k=6

list_means = random.sample(list_points,k)
list_means_pre = [(0.0,0.0)]*k
resume_iter = True 
   
#K-means beginning

t=0
while resume_iter:
    resume_iter = list_means != list_means_pre
    
#Attribution des points a un cluster

    list_sets = [[] for i in range(k)]   
    step_E(list_points,list_means,list_sets)        

#Recalcul des centres des clusters

    list_means_pre = list_means.copy()
    step_M(k,list_sets,list_means)
    
#Graphic part
            
    create_graph(k,list_sets)
    
    t+=1
    
