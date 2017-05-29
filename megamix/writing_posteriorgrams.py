# -*- coding: utf-8 -*-

"""
Created on Thu Mar  2 17:36:06 2017

:author: Elina Thibeau-Sutre
"""

import GMM
import VBGMM
import DPGMM

import numpy as np
import h5features as h5f
import h5py

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('method', help='The EM algorithm used')
    parser.add_argument('init', help='The method used to initialize')
    parser.add_argument('type_init', help='resp or mcw')
    parser.add_argument('file_name', help='the name of the file')
    args = parser.parse_args()
    
    method = args.method
    init = args.init
    type_init = args.type_init
    file_name = args.file_name
    
    data = h5f.Reader('/fhgfs/bootphon/scratch/ethibeau/mfcc_delta_cmn.features').read()
    items = data.items()
    labels = data.labels()
    features = data.features()
    points = np.concatenate(data.features(),axis=0)
    n_points,dim = points.shape
    
    
    if method == 'GMM':
        GM = GMM.GaussianMixture()
    elif method == 'VBGMM':
        GM = VBGMM.VariationalGaussianMixture()
    elif method == 'DPGMM':
        GM = DPGMM.DPVariationalGaussianMixture()
    else:
        raise ValueError("Invalid value for 'method' : %s "
                         "'method' should be in "
                         "['GMM','VBGMM','DPGMM']"
                         % method)

    #GMM
    directory = '/home/ethibeau-sutre/Results/' + method + '/' + init
    file_path = directory + '/' + method + '_' + type_init + '_' + file_name
    file = h5py.File(file_path)
    
    print(">>initialization")
    GM.read_and_init(file,points)
    
    # Writing posteriorgrams
    features_w = []
    for feat in features:
        log_resp = GM.predict_log_resp(feat)
        features_w.append(np.exp(log_resp))
    
    print(">>writing")
    print(directory)
    data_w = h5f.Data(items,labels,features_w)
    writer = h5f.Writer(directory + '/' + type_init + '_assignements.h5')
    writer.write(data_w)
    writer.close()