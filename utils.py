# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 14:48:20 2017

@author: Calixi
"""

"""
Created on Sun Feb 26 22:02:09 2017
@author: Calixi
"""

import pandas as pd
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

def read(file_name):
    fichier = pd.read_csv(file_name,sep = " ")
    matrix = fichier.as_matrix()
    
    return np.asarray(matrix)

def ellipses(covariance,mean):
      
    lambda_, v = np.linalg.eig(covariance)
    lambda_ = np.sqrt(lambda_)
    
    ell = Ellipse(xy=(mean[0], mean[1]),
              width=lambda_[0]*2, height=lambda_[1]*2,
              angle=np.rad2deg(np.arccos(v[0, 0])))
    ell.set_facecolor('none')
    ell.set_edgecolor('k')
    return ell

def ellipses_multidimensional(covariance,mean,d):
    
    lambda_, v = np.linalg.eig(covariance)
    lambda_sqrt = np.sqrt(lambda_)
    
    #We are looking for the extrema for each dimension
    principal_idx = np.argmax(abs(v[d] * lambda_sqrt))
    
    #This is done in order to not take the same previous vector
    possible_eig_vectors = np.copy(v[d+1])
    possible_eig_vectors[principal_idx] = 0
    principal_idx_2 = np.argmax(abs(possible_eig_vectors * lambda_sqrt))
    
    width = lambda_sqrt[principal_idx] * 2
    height = lambda_sqrt[principal_idx_2] * 2
    
#    Eigen values are ordered decreasingly
    if principal_idx > principal_idx_2:
        principal_idx = principal_idx_2
        width,height = height,width
    
    angle = np.rad2deg(np.arccos(v[d,principal_idx]))
    
    ell = Ellipse(xy=(mean[d], mean[d+1]),
              width=width, height=height,
              angle=angle)
    ell.set_facecolor('none')
    ell.set_edgecolor('k')
    return ell

if __name__ == '__main__':
    
    covariance = np.asarray([[10,0,0],[0,100,-20],[0,-20,10]])
    centre = np.asarray([0,0,0])
    
    dim = len(centre)
    
    for d in range(dim-1):
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ell = ellipses_multidimensional(covariance,centre,d)
        
        
        ax.add_artist(ell)
        ax.plot(centre[d],centre[d+1],'kx')
        
        
        plt.xticks(np.arange(-10,11,5))
        plt.yticks(np.arange(-10,11,5))
        
        titre = 'figure_proj_' + str(d)
        plt.savefig(titre)
        plt.close("all")
        
    covariance = np.asarray([[100,-20],[-20,10]])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ell = ellipses(covariance,centre)
    
    
    ax.add_artist(ell)
    ax.plot(centre[d],centre[d+1],'kx')
    
    
    plt.xticks(np.arange(-10,11,5))
    plt.yticks(np.arange(-10,11,5))
    
    titre = 'figure_proj_2'
    plt.savefig(titre)
    plt.close("all")