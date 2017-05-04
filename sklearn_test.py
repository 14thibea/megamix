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
import scipy.special
import GMM3

def convergence_criterion(n_components,points,log_resp,alpha_0,beta_0,nu_0,
                          alpha,beta,nu,cov,cov_prior,means,means_prior):
    
    resp = np.exp(log_resp)
    n_points,dim = points.shape
    
    # Convenient statistics
    N = np.exp(logsumexp(log_resp,axis=0)) + 10*np.finfo(resp.dtype).eps    #Array (n_components,)
    X_barre = np.tile(1/N, (dim,1)).T * np.dot(resp.T,points)               #Array (n_components,dim)
    S = np.zeros((n_components,dim,dim))                                    #Array (n_components,dim,dim)
    for i in range(n_components):
        diff = points - X_barre[i]
        diff_weighted = diff * np.tile(np.sqrt(resp[:,i:i+1]), (1,dim))
        S[i] = 1/N[i] * np.dot(diff_weighted.T,diff_weighted)
        
        S[i] += 1e-6 * np.eye(dim)
    
    inv_prec = cov * nu[:,np.newaxis,np.newaxis]
    inv_prec_init = cov_prior
    prec = np.linalg.inv(inv_prec)
    prec_init = np.linalg.inv(inv_prec_init)
    
    log_det_inv_prec = np.log(np.linalg.det(inv_prec))
    
    lower_bound = np.zeros(n_components)
    
    for i in range(n_components):
        
        log_weights_i = scipy.special.psi(alpha[i]) - scipy.special.psi(np.sum(alpha))
        
        digamma_sum = 0
        for j in range(dim):
            digamma_sum += scipy.special.psi((nu[i] - j)/2)
        log_det_prec_i = digamma_sum + dim * np.log(2) - log_det_inv_prec[i] #/!\ Inverse
        
        #First line
        lower_bound[i] = log_det_prec_i - dim/beta[i] - nu[i]*np.trace(np.dot(S[i],prec[i]))
        diff = X_barre[i] - means[i]
        lower_bound[i] += -nu[i]*np.dot(diff,np.dot(prec[i],diff.T)) - dim*np.log(2*np.pi)
        lower_bound[i] *= 0.5 * N[i]
        
        #Second line
        lower_bound[i] += (alpha_0 - alpha[i]) * log_weights_i
        lower_bound[i] += utils.log_B(prec_init,nu_0) - utils.log_B(prec[i],nu[i])
        
        resp_i = resp[:,i:i+1]
        log_resp_i = log_resp[:,i:i+1]
        
        lower_bound[i] += np.sum(resp_i) * log_weights_i - np.sum(resp_i*log_resp_i)
        lower_bound[i] += 0.5 * (nu_0 - nu[i]) * log_det_prec_i
        lower_bound[i] += dim*0.5*(np.log(beta_0) - np.log(beta[i]))
        lower_bound[i] += dim*0.5*(1 - beta_0/beta[i] + nu[i])
        
        #Third line without the last term which is not summed
        diff = means[i] - means_prior
        lower_bound[i] += -0.5*beta_0*nu[i]*np.dot(diff,np.dot(prec[i],diff.T))
        lower_bound[i] += -0.5*nu[i]*np.trace(np.dot(inv_prec_init,prec[i]))
            
    result = np.sum(lower_bound)
    result += utils.log_C(alpha_0 * np.ones(n_components))- utils.log_C(alpha)
    
    return result

def convergence_criterion_GM(points,log_resp,means,cov,covariance_type):
    """
    This method returns the log likelihood at the end of the k_means.
    
    @param points: an array of points (n_points,dim)
    @return: log likelihood measurement (float)
    """
    n_points = len(points)
    
    log_weights = logsumexp(log_resp, axis=0) - np.log(n_points)
    
    log_normal_matrix = GMM3._log_normal_matrix(points,means,cov,covariance_type)
    
    log_weights_duplicated = np.tile(log_weights, (n_points,1))
    log_product = log_normal_matrix + log_weights_duplicated
    log_product = logsumexp(log_product,axis=1)
    return np.sum(log_product)

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
    
    points_data = utils.read("../data/EMGaussienne.data")
    points_test = utils.read("../data/EMGaussienne.test")
    
    path = '../data/data.pickle'
    with open(path, 'rb') as fh:
        data = pickle.load(fh)
        
    N=1500
    k=100
    n_iter = 1000
        
    points = data['BUC']
    points_data = points[:N:]
    points_test = points_data
    
    n_points,dim = points_data.shape
    
    lower_bound_ = np.empty(n_iter)
    
    method = "DPGMM"
    covariance_type = 'full'
    for t in range(n_iter):
        print(t)
        print(">>predicting")
        GM = mixture.BayesianGaussianMixture(k,covariance_type=covariance_type,weight_concentration_prior_type='dirichlet_process')
#        GM = mixture.GaussianMixture(k,covariance_type=covariance_type,tol=1e-3)
        GM.fit(points_data)  
        print(">>fitted")
        _,log_assignements = GM._estimate_log_prob_resp(points_data)
#        _,means,cov,_ = GM._get_parameters()
        alpha,beta,means,nu,cov,_ = GM._get_parameters()
        beta_0 = GM.mean_precision_prior_
        alpha_0 = GM.weight_concentration_prior_
        nu_0 = GM.degrees_of_freedom_prior_
        cov_prior = GM.covariance_prior_
        means_prior = GM.mean_prior_
        if method == "DPGMM":
            alpha = alpha[0]
        print(">>parameters obtained")
    #    lower_bound_[t] = convergence_criterion_GM(points_data,log_assignements,means,cov,covariance_type)
        lower_bound_[t] = convergence_criterion(k,points_data,log_assignements,alpha_0,beta_0,nu_0,
                          alpha,beta,nu,cov,cov_prior,means,means_prior)        
    
#    log_weights = logsumexp(log_assignements, axis=0) - np.log(n_points)
#    weights = np.exp(log_weights)
#    
#    plt.title("Weights of the clusters")
#        
#    plt.hist(weights)
#    
#    directory = method + '/sklearn/'
#    titre = directory + '/figure_' + str(k) + "_" + covariance_type + "_weights.png"
#    plt.savefig(titre)
#    plt.close("all")


    plt.title("Lower bounds on " + str(n_iter) + " iterations")
    plt.hist(lower_bound_)
    
    directory = method + '/sklearn/'
    titre = directory + '/repartition_lower_bound_' + str(n_iter) + '_iter.png'
    plt.savefig(titre)
    plt.close("all")
    
    utils.write(directory + 'lower_bounds.csv',lower_bound_)