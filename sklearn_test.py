# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:33:23 2017

@author: Calixi
"""

from sklearn import mixture
import utils
import matplotlib.pyplot as plt
import os
import GMM2
import numpy as np
from scipy.misc import logsumexp

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
    
    k=5
    
    method = "DPGMM"
    for t in range(10):
        GM = mixture.BayesianGaussianMixture(t+1,covariance_type="full",weight_concentration_prior_type='dirichlet_process')
    
        GM.fit(points_data)
        assignements = GM.predict_proba(points_data)
        log_assignements = np.log(assignements)
        #        means,_,_ = GMM3.step_M(points,log_assignements,"full")
        #        cov = GM._get_covars()
        _,_,means,_,cov,_= GM._get_parameters()
        #    _,means,cov,_= GM._get_parameters()
        create_graph(points_data,means,cov,log_assignements,method,t)
        log_data_GMM2 = GMM2.log_likelihood(points_data,means,cov,assignements)
        log_data = GM.score_samples(points_data)
        log_test = GM.score_samples(points_test)
        print(log_data_GMM2)
        print(np.sum(log_data))
        print(np.sum(log_test))
        