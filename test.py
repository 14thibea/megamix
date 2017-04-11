# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:36:06 2017

@author: Calixi
"""

import utils
import Initializations as Init
import GMM3
import kmeans3

from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import pickle


from scipy.misc import logsumexp
import numpy as np

if __name__ == '__main__':
    
    k = 3
    
    full_covariance = True
    
    points = utils.read("D:/Mines/Cours/Stages/Stage_ENS/Code/Problem/data/EMGaussienne.data")
    means = Init.initialization_plus_plus(points,k)
    n_points,dim = points.shape
    
    print(means)
    
    assignements = kmeans3.step_E(points,means)
    
    for i in range(k):
        sets = assignements[:,i:i+1]
        n_sets = np.sum(sets)
        sets = points * np.tile(sets, (1,dim))
        means[i] = np.mean(sets, axis=0)*n_points/n_sets
    print(means)
    
    means2 = np.zeros((k,dim))
    
    for j in range(k):
        sets2 = [points[i] for i in range(n_points) if (assignements[i][j]==1)]
        means2[j] = np.mean(sets2, axis=0)
    
    print(means2)
        
    
    path = 'D:/Mines/Cours/Stages/Stage_ENS/Code/Problem/data/data.pickle'
    with open(path, 'rb') as fh:
        data = pickle.load(fh)
        
    
        
#    points = data['BUC']
#    points = points[0:500:]
#    points = points[:,0:2]
#
#    minima = points.min(axis=0)
#    maxima = points.max(axis=0)
#    diff = maxima - minima
#    print(diff)
#    cov_interval = np.power(diff/(10*k),2)
#    print(cov_interval)
#    
#    cov = np.diag(cov_interval)
#    mean = (maxima + minima)/2
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.plot(mean[0],mean[1],'kx')
#    ell = utils.ellipses(cov,mean)
#    ax.add_artist(ell)
#    ax.set_xticks((minima[0],maxima[0]))
#    ax.set_yticks((minima[1],maxima[1]))
#    
#    cov = np.tile(cov,(k,1,1))
#    print(cov)
    
    
    