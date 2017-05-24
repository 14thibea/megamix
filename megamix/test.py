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
import h5features as h5f

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

    N=1500
    k=100
    
    path = 'D:/Mines/Cours/Stages/Stage_ENS/Code/data/data.pickle'
    with open(path, 'rb') as fh:
        data = pickle.load(fh)
    
    
    points = data['BUC']
    points_data = points[:N:]
#    idx1 = np.random.randint(0,high=len(points),size=N)
#    points_data = points[idx1,:]
    
    GM = GMM.GaussianMixture(k,covariance_type="full",patience=0,tol=1e-3,reg_covar=1e-6)
#    GM = VBGMM.VariationalGaussianMixture(k,tol=1e-3,init="GMM")
#    GM = DPGMM.VariationalGaussianMixture(k,tol=1e-3,init='kmeans')

    #GMM
    print(">>predicting")
    GM.fit(feature1,draw_graphs=False)
    print()
    