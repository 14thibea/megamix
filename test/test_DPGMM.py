# -*- coding: utf-8 -*-
"""
Created on Fri May 19 17:35:18 2017

@author: Elina Thibeau-Sutre
"""
import numpy as np
from megamix import DPVariationalGaussianMixture
from megamix import base
from scipy.special import psi
from scipy.misc import logsumexp
import time
from joblib import Parallel,delayed

def empirical_covariances(points,mu,weight,resp,reg_covar):
    diff = points - mu
    diff_weighted = diff * resp
    cov = 1/weight * np.dot(diff_weighted.T,diff)
    cov.flat[::dim + 1] += reg_covar
    return cov

if __name__ == '__main__':
    
    n_points = 10000
    k = 100
    dim = 39
    
    log_prob = np.empty((n_points,k))
    data = np.random.randn(n_points,dim)
    resp = np.random.randn(n_points,k)
    
#    GM = DPVariationalGaussianMixture(k,init='random')
#    GM2 = DPVariationalGaussianMixture(k,init='kmeans')
#    GM3 = DPVariationalGaussianMixture(k,init='plus')
#    
#    GM.fit(data)
#    GM2.fit(data)
#    GM3.fit(data)
#    
    log_weights = np.empty(k)
#    
    GM = DPVariationalGaussianMixture(k,init='kmeans',type_init='mcw',n_jobs=3)
    t_ant = time.time()
    GM._initialize(data)
    
    t0 = time.time()
    GM._step_M(data,resp)
    t1 = time.time()
    N = np.sum(resp,axis=0) + 10*np.finfo(resp.dtype).eps
    t2 = time.time()
    X_barre = 1/N[:,np.newaxis] * np.dot(resp.T,data)
    t3 = time.time()
    S = np.zeros((k,dim,dim))
    for i in range(k):
        S[i] = empirical_covariances(data,X_barre[i],N[i],resp[:,i:i+1],GM.reg_covar)
    t4 = time.time()
    1/N[0]*np.dot(data.T,data)
    t5 = time.time()
    for i in range(k):
        if i==0:
            log_weights[i] = psi(GM.alpha[i][0]) - psi(np.sum(GM.alpha[i]))
        else:
            log_weights[i] = psi(GM.alpha[i][0]) - psi(np.sum(GM.alpha[i]))
            log_weights[i] += log_weights[i-1] + psi(GM.alpha[i-1][1]) - psi(GM.alpha[i-1][0])
    t6 = time.time()
    diff = data - X_barre[0]
    t7 = time.time()
    diff_weighted = diff * resp[:,i:i+1]
    t8 = time.time()
    log_resp = np.zeros((n_points,k))
    
        
    log_gaussian = base._log_normal_matrix(data,GM.means,GM.cov,'full')
    digamma_sum = np.sum(psi(.5 * (GM.nu - np.arange(0, dim)[:,np.newaxis])),0)
    log_lambda = digamma_sum + dim * np.log(2) + dim/GM.beta
    
    log_prob = GM.log_weights + log_gaussian + 0.5 * (log_lambda - dim * np.log(GM.nu))
    
    log_prob_norm = logsumexp(log_prob, axis=1)
    log_resp = log_prob - log_prob_norm[:,np.newaxis]
    t9 = time.time()
    S[0].flat[::dim + 1] += GM.reg_covar
    t10 = time.time()
    base._log_normal_matrix(data,GM.means,GM.cov,'full')
    t11 = time.time()
    precisions_chol = base._compute_precisions_chol(GM.cov,'full')
    t12 = time.time()
    for k, (mu, prec_chol) in enumerate(zip(GM.means,precisions_chol)):
        y = np.dot(data,prec_chol) - np.dot(mu,prec_chol)
        log_prob[:,k] = np.sum(np.square(y), axis=1)
    t13 = time.time()
    covariances = Parallel(n_jobs=3,backend='threading')(delayed(empirical_covariances)(data,X_barre[i],N[i],resp[:,i:i+1],GM.reg_covar)
    for i in range(k))
    t14 = time.time()
    res = GM._step_E(data)
    t15 = time.time()
    log_prob_norm = logsumexp(log_prob, axis=1)
    t16 = time.time()
    base._log_normal_matrix(data,GM.means,GM.cov,'full',3)
    t17 = time.time()
    log_det_chol = np.log(np.linalg.det(precisions_chol))
    t18 = time.time()
    result = -.5 * (dim * np.log(2*np.pi) + log_prob) + log_det_chol
    t19 = time.time()
                   
    print('initialization :',t0-t_ant)
    print('step M :', t1-t0)
    print('N : ', t2-t1)
    print('X_barre :', t3-t2)
    print('S :', t4-t3)
    print('S parallel :',t14-t13)
    print('simple dot :', (t5-t4)*k)
    print('poids :',t6-t5)
    print('soustract :',(t7-t6)*k)
    print('multiply :', (t8-t7)*k)
    print('flat :',(t10-t9)*k)
    print('step E :',t15-t14)
    print('step E original :',(t9-t8))
    print('boucle for :',(t13-t12))
    print('log normal matrix',(t11-t10))
    print('log normal matrix 3',(t17-t16))
    print('compute precisions chol :',(t12-t11))
    print('log det chol :',(t18-t17))
    print('logsumexp :',(t16-t15))
    print('result log_normal_matrix :',(t19-t18))
    
##    t0 = time.time()
#    res = base._log_normal_matrix(data,GM.means,GM.cov,'full')
#    t1 = time.time()
#    res_chol = base._compute_precisions_chol(GM.cov,'full')
#    t2 = time.time()
#    det_chol = np.linalg.det(res_chol)
#    t3 = time.time()
#    
#    for i, (mu, prec_chol) in enumerate(zip(GM.means,res_chol)):
#        y = np.dot(data,prec_chol) - np.dot(mu,prec_chol)
##        log_prob[:,i] = np.sum(np.square(y),axis=1)
#        
#    t4 = time.time()
#    
#    for i, (mu, prec_chol) in enumerate(zip(GM.means,res_chol)):
#        log_prob[:,i] = np.sum(np.square(y), axis=1)
#        
#    t5 = time.time()
#    GM._step_E(data)
#    t6 = time.time()
    
#    for i in range(k):
#        y = np.dot(data,res_chol[i]) - np.dot(GM.means[i],prec_chol)
#    
#    t7 = time.time()
#    
#    print('step E :', t6-t5)
#    print('log normal matrix :', t1-t0)
#    print(t2-t1,t3-t2,t4-t3,t5-t4)
#    print(t7-t6)
#    
    