# -*- coding: utf-8 -*-
"""
Created on Fri May 19 17:35:18 2017

@author: Elina Thibeau-Sutre
"""
import numpy as np
from megamix import Kmeans
import time
from megamix.initializations import initialization_plus_plus,initialization_AF_KMC
from megamix.kmeans import dist_matrix
from joblib import Parallel,delayed

def euclidean_distances(X,Y,XX=None,YY=None):
    
    if XX is None:
        XX = np.einsum('ij,ij->i', X, X)
    if YY is None:
        YY = np.sum(np.square(Y))
    XY = np.dot(X,Y)
    
    return XX - 2*XY + YY

def dist_matrix_for(X,Y):
    
    X_len,_ = X.shape
    XX = np.einsum('ij,ij->i', data, data)
    if len(Y.shape) > 1:    
        Y_len,Y_wid = Y.shape
        YY = np.sum(np.square(Y),axis=1)
        distances = np.zeros(shape=(X_len,Y_len),dtype = 'float64')    
        for i in range(Y_len):
            distances[:,i] = euclidean_distances(X,Y[i],XX,YY[i])
        return distances
            
    elif len(Y.shape)==1:
        YY = np.sum(np.square(Y))
        distances = euclidean_distances(X,Y,XX,YY)
        return distances
    
def dist_matrix_parallel(X,Y,n_jobs=1):
    
    X_len,_ = X.shape
    XX = np.einsum('ij,ij->i', data, data)
    if len(Y.shape) > 1:    
        Y_len,Y_wid = Y.shape
        YY = np.sum(np.square(Y),axis=1)
        distances = Parallel(n_jobs=n_jobs,backend='threading')(delayed(euclidean_distances)(X,Y[i],XX,YY[i]) for i in range(Y_len))    
        return np.asarray(distances).T
            
    elif len(Y.shape)==1:
        YY = np.sum(np.square(Y))
        distances = euclidean_distances(X,Y,XX,YY)
        return distances
        
if __name__ == '__main__':
    
    n_points = 10000
    k = 1000
    dim = 39
    n_iter = 100
    tol = 0
    
    means = np.empty((k,dim))
    log_prob = np.empty((n_points,k))
    data = np.random.randn(n_points,dim)
    test = np.random.randn(n_points,dim)
    
    KM = Kmeans(k,init='plus')
    KM.means = np.random.randn(k,dim)
    KM._is_initialized = True
#    t0 = time.time()
#    KM.fit(data,tol=tol)
#    t1 = time.time()
##    initialization_plus_plus(k,data)
#    t2 = time.time()
##    initialization_AF_KMC(k,data)
    t3 = time.time()
    res = KM._step_E(data)
    t4 = time.time()
    KM._step_M(data,res)
    t5 = time.time()
    KM.distortion(data,res)
    t6 = time.time()
    M = dist_matrix(data,KM.means)
    t7 = time.time()
    distances = np.zeros(shape=(n_points,k),dtype='float64')
    for i in range(k):
        distances[:,i] = np.linalg.norm(data-KM.means[i],axis=1)
    t8 = time.time()
    distances2 = Parallel(n_jobs=2,backend='threading')(delayed(np.linalg.norm)(data-KM.means[i],axis=1) for i in range(k))
    t9 = time.time()
    XX = np.einsum('ij,ij->i', data, data)
    YY = np.sum(np.square(KM.means),axis=1)
    distances3 = Parallel(n_jobs=2,backend='threading')(delayed(euclidean_distances)(data,KM.means[i],XX,YY[i]) for i in range(k))
    t10 = time.time()
    XX = np.einsum('ij,ij->i', data, data)
    YY = np.einsum('ij,ij->i', KM.means, KM.means)
    distances3 = Parallel(n_jobs=2,backend='threading')(delayed(euclidean_distances)(data,KM.means[i],XX,YY[i]) for i in range(k))
    t10bis = time.time()
    X = data
    Y = KM.means[0] 
    XX = np.einsum('ij,ij->i', X, X)
    t11 = time.time()
    YY = np.sum(np.square(Y))
    t12 = time.time()
    XY = np.dot(X,Y)
    t13 = time.time()
    YY2 = np.sum(np.square(KM.means),axis=1)
    t14 = time.time()
    XX2 = np.sum(np.square(data),axis=1)
    t15 = time.time()
    XX3 = np.sum(data*data,axis=1)
    t16 = time.time()
    M2 = dist_matrix_for(data,KM.means)
    t17 = time.time()
    M3 = dist_matrix_for(data,KM.means[0])
    t18 = time.time()
    Mpar2 = dist_matrix_parallel(data,KM.means)
    t19 = time.time()
    Mpar3 = dist_matrix_parallel(data,KM.means[0])
    t20 = time.time()
    
#    print('kmeans :',t1-t0)
#    print('plus_plus :',t2-t1)
#    print('AF_KMC :',t3-t2)
    print('step E :',t4-t3)
    print('step M :',t5-t4)
    print('dist_matrix sk:',t7-t6)
    print('dist_matrix for:',t8-t7)
    print('dist_matrix parallel :',t9-t8)
    print('tuning dist_matrix :',t10-t9)
    print('tuning dist matrix :',t10bis - t10)
    print('XX :',(t11-t10)*k)
    print('YY :',(t12-t11)*k)
    print('XY :',(t13-t12)*k)
    print('other YY :',t14-t13)
    print('other XX :',t15-t14)
    print('other XX :',t16-t15)
    print('dist matrix for :',t17-t16)
    print('try with vector :',t18-t17)
    print('dist matrix par :',t19-t18)
    print('try with vector :',t20-t19)