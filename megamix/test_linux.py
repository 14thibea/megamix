# -*- coding: utf-8 -*-

"""
Created on Thu Mar  2 17:36:06 2017

:author: Elina Thibeau-Sutre
"""

import GMM
import VBGMM
import DPGMM
import utils

import matplotlib.pyplot as plt
import pickle
import numpy as np
import h5features

def matrix_sym(matrix):
    sym = True
    n_rows,_ = matrix.shape
    for i in range(n_rows):
        for j in range(n_rows):
            if matrix[i,j] != matrix[j,i]:
                sym = False
    return sym

def create_hist(filename,data,title=""):
    plt.title(title)
    plt.hist(data)
    
    plt.savefig(filename)
    plt.close("all")
    
if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('early_stop', help='True if early stop is computed')
    parser.add_argument('method', help='The EM algorithm used')
    parser.add_argument('type_init', help='How the algorithm will be initialized ("resp" or "mcw")')
    parser.add_argument('init', help='The method used to initialize')
    parser.add_argument('covariance_type', help='the covariance type : "full" or "spherical"')
    parser.add_argument('cluster_number', help='the number of clusters wanted')
    args = parser.parse_args()
    
    N=15000
    n_iter = 1
    early_stop = True
    k=int(args.cluster_number)
    method = args.method
    init=args.init
    type_init = args.type_init
    covariance_type = args.covariance_type
    
    path = '../../data/data.pickle'
    with open(path, 'rb') as fh:
        data = pickle.load(fh)
        
    points = data['BUC']
    n_points,_ = points.shape
    if early_stop:
        idx = np.random.randint(0,high=len(points),size=N)
        points_data = points[idx,:]
        idx = np.random.randint(0,high=len(points),size=N)
        points_test = points[idx,:]
    else:
        points_data = points[:N:]
        points_test = None
    
    
#    data = h5features.Reader('../../data/mfcc_delta_cmn.features').read()
    data = h5features.Reader('/fhgfs/bootphon/scratch/ethibeau/mfcc_delta_cmn.features').read()
    points = np.concatenate(data.features(),axis=0)
    n_points,dim = points.shape
    
    if early_stop:
        np.random.shuffle(points)
        points_data = points[:n_points//2:]
        points_test = points[n_points//2::]
    else:
        points_data = points[:n_points//2:]
        points_test = None
    
    lower_bound = np.arange(n_iter)
    
    if method == 'GMM':
        GM = GMM.GaussianMixture(k,init=init,type_init=type_init)
    elif method == 'VBGMM':
        GM = VBGMM.VariationalGaussianMixture(k,init=init,type_init=type_init)
    elif method == 'DPGMM':
        GM = DPGMM.DPVariationalGaussianMixture(k,init=init,type_init=type_init)
    else:
        raise ValueError("Invalid value for 'method' : %s "
                         "'method' should be in "
                         "['GMM','VBGMM','DPGMM']"
                         % method)

    #GMM
    directory = 'D:/Mines/Cours/Stages/Stage_ENS/Code/Results/' + method + '/' + init
    
    for i in range(n_iter):
        print(i)
        print(">>predicting")
        GM.fit(points_data,points_test,saving='log_iter',directory=directory,legend='_mfcc_delta_cmn')
        lower_bound[i] = GM.convergence_criterion_data[-1]
        print()
    
    directory = 'D:/Mines/Cours/Stages/Stage_ENS/Code/Results/' + method + '/' + init
    print('early stop : ' + str(early_stop))
    print(directory)
    utils.write(directory + '/lower_bounds_early_stop_' + str(early_stop) + '.csv',lower_bound)