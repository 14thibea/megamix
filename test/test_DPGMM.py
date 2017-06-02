# -*- coding: utf-8 -*-
"""
Created on Fri May 19 17:35:18 2017

@author: Elina Thibeau-Sutre
"""
import numpy as np
from megamix import DPVariationalGaussianMixture
from megamix import base
from scipy.special import psi
import time

if __name__ == '__main__':
    
    n_points = 10000
    k = 1000
    dim = 39
    
    log_prob = np.empty((n_points,k))
    data = np.random.randn(n_points,dim)
    resp = np.random.randn(n_points,k)
    
    GM = DPVariationalGaussianMixture(k,init='random')
#    GM2 = DPVariationalGaussianMixture(k,init='kmeans')
#    GM3 = DPVariationalGaussianMixture(k,init='plus')
#    
    GM.fit(data)
#    GM2.fit(data)
#    GM3.fit(data)
#    
#    log_weights = np.empty(k)
#    
#    GM = DPVariationalGaussianMixture(k,init='random',type_init='mcw')
#    GM._initialize(data)
#    
#    t0 = time.time()
#    GM._step_M(data,resp)
#    t1 = time.time()
#    N = np.sum(resp,axis=0) + 10*np.finfo(resp.dtype).eps
#    t2 = time.time()
#    X_barre = 1/N[:,np.newaxis] * np.dot(resp.T,data)
#    t3 = time.time()
#    S = np.zeros((k,dim,dim))
#    for i in range(k):
##        diff = data - X_barre[i][np.newaxis:,]
##        diff = data - X_barre[i]
##        diff_weighted = diff * resp[:,i:i+1]
#        S[i] = 1/N[i] * np.dot(data.T,data)
#        
##        S[i] += GM.reg_covar * np.eye(dim)
#    t4 = time.time()
##    diff = data - X_barre[i]
##    diff_weighted = diff * resp[:,i:i+1]
#    1/N[0]*np.dot(data.T,data)
##    S[i] += GM.reg_covar * np.eye(dim)
#    t5 = time.time()
#    for i in range(k):
#        if i==0:
#            log_weights[i] = psi(GM.alpha[i][0]) - psi(np.sum(GM.alpha[i]))
#        else:
#            log_weights[i] = psi(GM.alpha[i][0]) - psi(np.sum(GM.alpha[i]))
#            log_weights[i] += log_weights[i-1] + psi(GM.alpha[i-1][1]) - psi(GM.alpha[i-1][0])
#    t6 = time.time()
#    data - X_barre[0]
#    t7 = time.time()
#    
#    print('step M :', t1-t0)
#    print('N : ', t2-t1)
#    print('X_barre :', t3-t2)
#    print('S :', t4-t3)
#    print('simple dot :', (t5-t4)*k)
#    print('poids :',t6-t5)
#    print('soustract :',(t7-t6)*k)
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
    