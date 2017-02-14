# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 17:22:14 2017

@author: Calixi
"""

import csv
import random
import matplotlib
from scipy.spatial import distance

#Lecture du fichier

fichier = open("D:/Mines/Cours/Stages/Stage_ENS/Problem/EMGaussienne.data",'r')
reader = csv.reader(fichier,delimiter=" ")
list_points = []
for line in reader:
    a,b = line
    point = (float(a),float(b))
    list_points.append(point)
fichier.close()

#k-means

#initialisation

k=3
list_moyenne = random.sample(list_points,k)

list_sets = []
for i in range(k):
    list_sets.append([])

#Attribution des points a un cluster
for point in list_points:
    mean_number = 0
    cluster_number = 0
    minimum = 100
    for mean in list_moyenne:
        dst = distance.euclidean(mean,point)
        if (dst<minimum):
            minimum = dst
            cluster_number = mean_number
        mean_number+=1
    list_sets[cluster_number].append(point)

#Recalcul des centres des clusters

x=0
y=0

for i in range(k):
    length = len(list_sets[i])
    while(len(list_sets[i])!=0):
        point = list_sets[i].pop()
        x+=point[0]/length
        y+=point[1]/length
    list_moyenne[i] = (x,y)
    
#Reattribution des points
    
for point in list_points:
    mean_number = 0
    cluster_number = 0
    minimum = 100
    for mean in list_moyenne:
        dst = distance.euclidean(mean,point)
        if (dst<minimum):
            minimum = dst
            cluster_number = mean_number
        mean_number+=1
    list_sets[cluster_number].append(point)

#Mise en forme pour pyploy

x_points = []
for i in range(k):
    x_points.append([])
    
y_points = []
for i in range(k):
    y_points.append([])
    
for i in range(k):
    while(len(list_sets[i])!=0):
        point = list_sets[i].pop()
        x_points[i].append(point[0])
        y_points[i].append(point[1])
        
matplotlib.pyplot.plot(x_points[0],y_points[0],'ro',x_points[1],y_points[1],'bo',x_points[2],y_points[2],'go')
