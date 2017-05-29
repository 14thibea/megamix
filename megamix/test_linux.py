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
    parser.add_argument('--early_stop', action='store_true', help='True if early stop is computed')
    parser.add_argument('method', help='The EM algorithm used')
    parser.add_argument('type_init', help='How the algorithm will be initialized ("resp" or "mcw")')
    parser.add_argument('init', help='The method used to initialize')
    parser.add_argument('covariance_type', help='the covariance type : "full" or "spherical"')
    parser.add_argument('cluster_number', help='the number of clusters wanted')
    args = parser.parse_args()
    
    early_stop = args.early_stop
    k=int(args.cluster_number)
    method = args.method
    init=args.init
    type_init = args.type_init
    covariance_type = args.covariance_type
    
    data = h5features.Reader('/fhgfs/bootphon/scratch/ethibeau/mfcc_delta_cmn.features').read()
    items = data.items()
    labels = data.labels()
    features = data.features()
    points = np.concatenate(data.features(),axis=0)
    n_points,dim = points.shape
    
    if early_stop:
        np.random.shuffle(points)
        points_data = points[:n_points//2:]
        points_test = points[n_points//2::]
    else:
        points_data = points
        points_test = None
    
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
        
    if early_stop:
        legend = '_mfcc_delta_cmn_early_stop'
    else:
        legend = '_mfcc_delta_cmn'

    #GMM
    directory = '/home/ethibeau-sutre/Results/' + method + '/' + init
    
    print(">>predicting")
    GM.fit(points_data,points_test,saving='log',directory=directory,legend=legend)
    print()
        
    if GM.early_stopping:
        print('you used early stopping')
    print('early stop : ', early_stop)
    print(directory)
    
    # Writing posteriorgrams
    features_w = []
    for feat in features:
        log_resp = GM.predict_log_resp(feat)
        features_w.append(np.exp(log_resp))
    
    data_w = h5features.Data(items,labels,features_w)
    writer = h5features.Writer(directory + '/' + type_init + '_assignements.h5')
    writer.write(data)
    writer.close()