# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:33:23 2017

@author: Calixi
"""

from sklearn import mixture
import utils
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.misc import logsumexp
import pickle

def create_graph(points,means,cov,log_assignements,method,t):
    """
    This method draws a 2D graph displaying the clusters and their means and saves it as a PNG file.
    If points have more than two coordinates, then it will be a projection including only the first coordinates.
    
    @param points: an array of points (n_points,dim)
    @param means: an array of k points which are the means of the clusters (n_components,dim)
    @param log_assignements: an array containing the log of soft assignements of every point (n_points,n_components)
    @full_covariance: a string in ['full','spherical']
    @param t: the figure number
    """
    
    k=len(means)
    n_points,dim = points.shape

    dir_path = method + '/sklearn/'
    directory = os.path.dirname(dir_path)

    try:
        os.stat(directory)
    except:
        os.mkdir(directory)  
    
    log_data = np.sum(GM.score_samples(points_data))
    
    x_points = [[] for i in range(k)]
    y_points = [[] for i in range(k)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("log likelihood = " + str(log_data) + " , k = " + str(k))
    
    
    log_prior_prob = logsumexp(log_assignements, axis=0) - np.log(n_points)
    
    for i in range(k):
        
        if log_prior_prob[i] > -4 :
        
            covariance = cov[i]
            
            ell = utils.ellipses(covariance,means[i])
            ax.add_artist(ell)
            x_points[i] = [points[j][0] for j in range(n_points) if (np.argmax(log_assignements[j])==i)]        
            y_points[i] = [points[j][1] for j in range(n_points) if (np.argmax(log_assignements[j])==i)]
    
            ax.plot(x_points[i],y_points[i],'o',alpha = 0.2)
            ax.plot(means[i][0],means[i][1],'kx')
            
    titre = directory + '/figure_sklearn_' + str(t)
    plt.savefig(titre)
    plt.close("all")
    
if __name__ == '__main__':
    
    points_data = utils.read("D:/Mines/Cours/Stages/Stage_ENS/Code/data/EMGaussienne.data")
    points_test = utils.read("D:/Mines/Cours/Stages/Stage_ENS/Code/data/EMGaussienne.test")
    
    path = 'D:/Mines/Cours/Stages/Stage_ENS/Code/data/data.pickle'
    with open(path, 'rb') as fh:
        data = pickle.load(fh)
        
    N=1500
        
    points = data['BUC']
    points_data = points[:N:]
    points_test = points_data
    
    _,dim = points_data.shape
    
    k=100
    
    method = "VBGMM"
    covariance_type = 'full'
#    for t in range(10):
    GM = mixture.BayesianGaussianMixture(k,covariance_type=covariance_type,max_iter=500,tol=1e-20,weight_concentration_prior_type='dirichlet_distribution')

    GM.fit(points_data)
    
    assignements = GM.predict_proba(points_data)
    weights = np.sum(assignements, axis=0)
    weights /= np.sum(weights)
    
    plt.title("Weights of the clusters")
        
    plt.hist(weights)
    
    directory = method + '/sklearn/'
    titre = directory + '/figure_' + str(k) + "_" + covariance_type + "_weights.png"
    plt.savefig(titre)
    plt.close("all")