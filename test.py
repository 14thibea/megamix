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

    N=1500
    k=100
    n_iter = 1000
    
    path = 'D:/Mines/Cours/Stages/Stage_ENS/Code/data/data.pickle'
    with open(path, 'rb') as fh:
        data = pickle.load(fh)
        
    points = data['BUC']
    points_data = points[:N:]
#    idx1 = np.random.randint(0,high=len(points),size=N)
#    points_data = points[idx1,:]
    
    lower_bound = np.arange(n_iter)
    
    GM = GMM.GaussianMixture(k,covariance_type="full",patience=0,tol=1e-4,reg_covar=1e-6)
#    GM = VBGMM.VariationalGaussianMixture(k,tol=1e-3,init="GMM")
#    GM = DPGMM.VariationalGaussianMixture(k,tol=1e-3,init='kmeans')

    #GMM
    for i in range(n_iter):
        print(i)
        print(">>predicting")
#        log_assignements_data,log_assignements_test = GMM.predict_log_assignements(points_data,points_test)
        log_assignements_data = GM.predict_log_assignements(points_data,draw_graphs=False)
        lower_bound[i] = GM.convergence_criterion_data[-1]
        print()
            
    plt.title("Lower bounds on " + str(n_iter) + " iterations")
    plt.hist(lower_bound)
    
    directory = GM.create_path()
    titre = directory + '/repartition_lower_bound_' + str(n_iter) + '_iter.png'
    plt.savefig(titre)
    plt.close("all")
    
    utils.write(directory + '/lower_bounds.csv',lower_bound)