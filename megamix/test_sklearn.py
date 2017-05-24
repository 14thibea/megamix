# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:33:23 2017

:author: Elina Thibeau-Sutre
"""

from sklearn import mixture
import utils
import os
import numpy as np
from scipy.misc import logsumexp
import pickle
from scipy.special import betaln,psi
import GMM

def convergence_criterion_VB(n_components,points,log_resp,alpha_0,beta_0,nu_0,
                             alpha,beta,nu,cov,cov_prior,means,means_prior):
        
    resp = np.exp(log_resp)
    n_points,dim = points.shape
    
    inv_prec = cov * nu[:,np.newaxis,np.newaxis]
    log_det_inv_prec = np.log(np.linalg.det(inv_prec))
    inv_prec_prior = cov_prior
    
    # Convenient statistics
    N = np.exp(logsumexp(log_resp,axis=0)) + 10*np.finfo(resp.dtype).eps    #Array (n_components,)
    X_barre = np.tile(1/N, (dim,1)).T * np.dot(resp.T,points)               #Array (n_components,dim)
    S = np.zeros((n_components,dim,dim))                                    #Array (n_components,dim,dim)
    for i in range(n_components):
        diff = points - X_barre[i]
        diff_weighted = diff * np.tile(np.sqrt(resp[:,i:i+1]), (1,dim))
        S[i] = 1/N[i] * np.dot(diff_weighted.T,diff_weighted)
        
        S[i] += 1e-6 * np.eye(dim)
    
    prec = np.linalg.inv(inv_prec)
    prec_prior = np.linalg.inv(inv_prec_prior)
    
    lower_bound = np.zeros(n_components)
    
    for i in range(n_components):
        
        log_weights_i = psi(alpha[i]) - psi(np.sum(alpha))
        
        digamma_sum = 0
        for j in range(dim):
            digamma_sum += psi((nu[i] - j)/2)
        log_det_prec_i = digamma_sum + dim * np.log(2) - log_det_inv_prec[i] #/!\ Inverse
        
        #First line
        lower_bound[i] = log_det_prec_i - dim/beta[i] - nu[i]*np.trace(np.dot(S[i],prec[i]))
        diff = X_barre[i] - means[i]
        lower_bound[i] += -nu[i]*np.dot(diff,np.dot(prec[i],diff.T))
        lower_bound[i] *= 0.5 * N[i]
        
        #Second line
        lower_bound[i] += (alpha_0 - alpha[i]) * log_weights_i
        lower_bound[i] += utils.log_B(prec_prior,nu_0) - utils.log_B(prec[i],nu[i])
        
        resp_i = resp[:,i:i+1]
        log_resp_i = log_resp[:,i:i+1]
        
        lower_bound[i] += np.sum(resp_i) * log_weights_i - np.sum(resp_i*log_resp_i)
        lower_bound[i] += 0.5 * (nu_0 - nu[i]) * log_det_prec_i
        lower_bound[i] += dim*0.5*(np.log(beta_0) - np.log(beta[i]))
        lower_bound[i] += dim*0.5*(1 - beta_0/beta[i] + nu[i])
        
        #Third line without the last term which is not summed
        diff = means[i] - means_prior
        lower_bound[i] += -0.5*beta_0*nu[i]*np.dot(diff,np.dot(prec[i],diff.T))
        lower_bound[i] += -0.5*nu[i]*np.trace(np.dot(inv_prec_prior,prec[i]))
            
    result = np.sum(lower_bound)
    result += utils.log_C(alpha_0 * np.ones(n_components))- utils.log_C(alpha)
    result -= n_points * dim * 0.5 * np.log(2*np.pi)
    
    return result

def convergence_criterion_DP(n_components,points,log_resp,alpha_0,beta_0,nu_0,
                             alpha,beta,nu,cov,cov_prior,means,means_prior):
    resp = np.exp(log_resp)
    n_points,dim = points.shape
    
    inv_prec = cov * nu[:,np.newaxis,np.newaxis]
    log_det_inv_prec = np.log(np.linalg.det(inv_prec))
    inv_prec_prior = cov_prior
        
    prec = np.linalg.inv(inv_prec)
    prec_prior = np.linalg.inv(inv_prec_prior)
    
    # Convenient statistics
    N = np.exp(logsumexp(log_resp,axis=0)) + 10*np.finfo(resp.dtype).eps    #Array (n_components,)
    X_barre = np.tile(1/N, (dim,1)).T * np.dot(resp.T,points)               #Array (n_components,dim)
    S = np.zeros((n_components,dim,dim))                                    #Array (n_components,dim,dim)
    for i in range(n_components):
        diff = points - X_barre[i]
        diff_weighted = diff * np.tile(np.sqrt(resp[:,i:i+1]), (1,dim))
        S[i] = 1/N[i] * np.dot(diff_weighted.T,diff_weighted)
        
        S[i] += 1e-6 * np.eye(dim)
    
    prec = np.linalg.inv(inv_prec)
    prec_prior = np.linalg.inv(inv_prec_prior)
    
    lower_bound = np.zeros(n_components)
    
    for i in range(n_components):
        
        digamma_sum = 0
        for j in range(dim):
            digamma_sum += psi((nu[i] - j)/2)
        log_det_prec_i = digamma_sum + dim * np.log(2) - log_det_inv_prec[i] #/!\ Inverse
        
        #First line
        lower_bound[i] = log_det_prec_i - dim/beta[i] - nu[i]*np.trace(np.dot(S[i],prec[i]))
        diff = X_barre[i] - means[i]
        lower_bound[i] += -nu[i]*np.dot(diff,np.dot(prec[i],diff.T))
        lower_bound[i] *= 0.5 * N[i]
        
        #Second line
        lower_bound[i] += utils.log_B(prec_prior,nu_0) - utils.log_B(prec[i],nu[i])
        
        resp_i = resp[:,i:i+1]
        log_resp_i = log_resp[:,i:i+1]
        
        lower_bound[i] -=  np.sum(resp_i*log_resp_i)
        lower_bound[i] += 0.5 * (nu_0 - nu[i]) * log_det_prec_i
        lower_bound[i] += dim*0.5*(np.log(beta_0) - np.log(beta[i]))
        lower_bound[i] += dim*0.5*(1 - beta_0/beta[i] + nu[i])
        
        #Third line without the last term which is not summed
        diff = means[i] - means_prior
        lower_bound[i] += -0.5*beta_0*nu[i]*np.dot(diff,np.dot(prec[i],diff.T))
        lower_bound[i] += -0.5*nu[i]*np.trace(np.dot(inv_prec_prior,prec[i]))
        
        #Terms with alpha
        lower_bound[i] += (N[i] + 1 - alpha[0][i]) * (psi(alpha[0][i] + alpha[1][i]))
        lower_bound[i] += (np.sum(N[i+1::]) + alpha_0 - alpha[1][i]) * (psi(alpha[1][i]) - psi(alpha[0][i] + alpha[1][i]))
        
            
    result = np.sum(lower_bound)
    result -= n_components * betaln(1,alpha_0)
    result += np.sum(betaln(alpha[0],alpha[1]))
    result -= n_points * dim * 0.5 * np.log(2*np.pi)
    
    return result

def convergence_criterion_GM(points,log_resp,means,cov,covariance_type='full'):
    """
    This method returns the log likelihood at the end of the k_means.
    
    @param points: an array of points (n_points,dim)
    @return: log likelihood measurement (float)
    """
    n_points = len(points)
    
    log_weights = logsumexp(log_resp, axis=0) - np.log(n_points)
    
    log_normal_matrix = GMM._log_normal_matrix(points,means,cov,covariance_type)
    
    log_weights_duplicated = np.tile(log_weights, (n_points,1))
    log_product = log_normal_matrix + log_weights_duplicated
    log_product = logsumexp(log_product,axis=1)
    
    return np.sum(log_product)

if __name__ == '__main__':
    
#    import argparse
#    parser = argparse.ArgumentParser()
#    parser.add_argument('method', help='The EM algorithm used')
#    parser.add_argument('cluster_number', help='the number of clusters wanted')
#    args = parser.parse_args()
    
    path = '../../data/data.pickle'
    with open(path, 'rb') as fh:
        data = pickle.load(fh)
        
    N=15000
    k=100
    n_iter = 1
    
    points = data['BUC']
    points_data = points[:N:]
    points_test = points_data
    
    n_points,dim = points_data.shape
    
    lower_bound_ = np.empty(n_iter)
    
#    method = args.method
    method = 'DPGMM'
    if method=='DPGMM':
        weight_concentration_prior_type='dirichlet_process'
        GM = mixture.BayesianGaussianMixture(k,weight_concentration_prior_type=weight_concentration_prior_type)
        
    elif method=='VBGMM':
        weight_concentration_prior_type='dirichlet_distribution'
        GM = mixture.BayesianGaussianMixture(k,weight_concentration_prior_type=weight_concentration_prior_type)
        
    elif method=='GMM':
        GM = mixture.GaussianMixture(k)
        
#    else:
#        raise ValueError("Invalid value for 'method' : %s "
#                         "'method' should be in "
#                         "['GMM','VBGMM','DPGMM']"
#                         % args.method)
    
    for t in range(n_iter):
        print(t)
        print(">>predicting")
        GM.fit(points_data)
        _,log_assignements = GM._estimate_log_prob_resp(points_data)
        
        if method == 'GMM':
            _,means,cov,_ = GM._get_parameters()
            lower_bound_[t] = convergence_criterion_GM(points_data,log_assignements,means,cov)
            
        else:
            alpha,beta,means,nu,cov,_ = GM._get_parameters()
            beta_0 = GM.mean_precision_prior_
            alpha_0 = GM.weight_concentration_prior_
            nu_0 = GM.degrees_of_freedom_prior_
            cov_prior = GM.covariance_prior_
            means_prior = GM.mean_prior_
            if method == 'VBGMM':
                lower_bound_[t] = convergence_criterion_VB(k,points_data,log_assignements,alpha_0,beta_0,nu_0,
                                                           alpha,beta,nu,cov,cov_prior,means,means_prior)
            elif method == 'DPGMM':
                lower_bound_[t] = convergence_criterion_DP(k,points_data,log_assignements,alpha_0,beta_0,nu_0,
                                                           alpha,beta,nu,cov,cov_prior,means,means_prior)
            
    
#    directory = '/home/ethibeau-sutre/Scripts/' + method + '/sklearn/'
#    utils.write(directory + 'lower_bounds.csv',lower_bound_)