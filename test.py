# -*- coding: utf-8 -*-

"""
Created on Thu Mar  2 17:36:06 2017

@author: Calixi
"""

import utils
import Initializations as Init
import GMM2
import GMM3
import kmeans3
import VBGMM

from sklearn.metrics.pairwise import euclidean_distances
from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pickle
import os
import scipy.special


from scipy.misc import logsumexp
import numpy as np

def matrix_sym(matrix):
    sym = True
    n_rows,_ = matrix.shape
    for i in range(n_rows):
        for j in range(n_rows):
            if matrix[i,j] != matrix[j,i]:
                sym = False
    return sym

def plot_comparison_means(means1, means2, legend):
    """
    Used to compare the models obtained with 2 different models
    """
    n_components = len(means1)
    if n_components != len(means2):
        raise ValueError("les deux modèles doivent utiliser"
                         " le même nombre de composants")
    
    sum_diff = 0
    for i in range(n_components):
        diff = np.linalg.norm(means1 - means2[i], axis=1)
        sum_diff += np.min(diff)
    
    means = np.concatenate((means1,means2),axis=0)
    
    mds = manifold.MDS(2)
    norm = np.linalg.norm(means,axis=1)
    means_normed = means / norm[:,np.newaxis]
    
    coord = mds.fit_transform(means_normed)
    stress = mds.stress_
    
    first_coords = coord[0:n_components:]
    second_coords = coord[n_components::]
    
    plt.plot(first_coords.T[0],first_coords.T[1],'bo')
    plt.plot(second_coords.T[0],second_coords.T[1],'yo')
    
    plt.title("stress = " + str(stress) + " sum of the differences = " + str(sum_diff))
    
    titre = 'figure_' + str(n_components) + "_means_comparison_" + legend + ".png"
    plt.savefig(titre)
    plt.close("all")

            

if __name__ == '__main__':
    
    k = 6

    points_data = utils.read("D:/Mines/Cours/Stages/Stage_ENS/Code/data/EMGaussienne.data")

#    means,covariance = Init.initialization_GMM(k,points_data,points_data)
    
    GMM = GMM3.GaussianMixture(k)
    GMM.set_parameters(means=np.asarray([[0,1]]))
    
    
    
    
    
    
        
    
    
    