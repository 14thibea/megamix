# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:33:23 2017

@author: Calixi
"""

from sklearn import mixture
import CSVreader
import matplotlib.pyplot as plt

def draw(k,labels,points,t):
    
    nb_points = len(points)
    
    x_points = [[] for i in range(k)]
    y_points = [[] for i in range(k)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for i in range(k):
        x_points[i] = [points[j][0] for j in range(nb_points) if (labels[j]==i)]        
        y_points[i] = [points[j][1] for j in range(nb_points) if (labels[j]==i)]
    
        ax.plot(x_points[i],y_points[i],'o')
        
    titre = 'figure_' + str(t)
    plt.savefig(titre)
    plt.close("all")
    

if __name__ == '__main__':
    
    k = 4
    N = 500
    
    points = CSVreader.read("D:/Mines/Cours/Stages/Stage_ENS/Code/Problem/EMGaussienne.data")
    points = points[:,0:-1]
    
    points2 = CSVreader.read("D:/Mines/Cours/Stages/Stage_ENS/Code/Problem/EMGaussienne.test")
    points2 = points2[:,0:-1]
    
    points = points[0:N:]
    
    GM = mixture.GaussianMixture(n_components=k,tol=0.00001,init_params='random')
    GM.fit(points)
    labels = GM.predict(points)
    draw(k,labels,points,0)