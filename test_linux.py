# -*- coding: utf-8 -*-

"""
Created on Thu Mar  2 17:36:06 2017

@author: Calixi
"""

import GMM
import VBGMM
import DPGMM
import utils

import matplotlib.pyplot as plt
import pickle
import numpy as np

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
    early_stop = False
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('method', help='The EM algorithm used')
    parser.add_argument('type_init', help='How the algorithm will be initialized ("resp" or "mcw")')
    parser.add_argument('init', help='The method used to initialize')
    parser.add_argument('covariance_type', help='the covariance type : "full" or "spherical"')
    parser.add_argument('cluster_number', help='the number of clusters wanted')
    args = parser.parse_args()
    
    N=15000
    k=int(args.cluster_number)
    n_iter = 1000
    
    path = '/home/ethibeau-sutre/data/data.pickle'
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
    
    
    lower_bound = np.arange(n_iter)
    
    if args.method == 'GMM':
        GM = GMM.GaussianMixture(k,covariance_type=args.covariance_type,type_init=args.type_init)
    elif args.method == 'VBGMM':
        GM = VBGMM.VariationalGaussianMixture(k,init=args.init,type_init=args.type_init)
    elif args.method == 'DPGMM':
        GM = DPGMM.VariationalGaussianMixture(k,init=args.init,type_init=args.type_init)
    else:
        raise ValueError("Invalid value for 'method' : %s "
                         "'method' should be in "
                         "['GMM','VBGMM','DPGMM']"
                         % args.method)

    #GMM
    for i in range(n_iter):
        print(i)
        print(">>predicting")
        if early_stop:
            log_assignements_data,log_assignements_test = GMM.predict_log_assignements(points_data,points_test)
        else:
            log_assignements_data = GM.predict_log_assignements(points_data,draw_graphs=False)
        lower_bound[i] = GM.convergence_criterion_data[-1]
        print()
    
    directory = GM.create_path()
    print(directory)
    utils.write(directory + '/lower_bounds_' + args.type_init + '.csv',lower_bound)