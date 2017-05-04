# -*- coding: utf-8 -*-

"""
Created on Thu Mar  2 17:36:06 2017

@author: Calixi
"""

import GMM3
import VBGMM2
import DPGMM2
import utils

from sklearn import manifold
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

def plot_comparison_means(means1, means2, legend):
    """
    Used to compare the models obtained with 2 different models
    """
    n_components = len(means1)
    if n_components != len(means2):
        raise ValueError("les deux modèles doivent utiliser"
                         " le même nombre de composants")
    
    sum_diff = 0
    for i in range(n_components):
        diff = np.linalg.norm(means1 - means2[i], axis=1)
        sum_diff += np.min(diff)
    
    means = np.concatenate((means1,means2),axis=0)
    
    mds = manifold.MDS(2)
    norm = np.linalg.norm(means,axis=1)
    means_normed = means / norm[:,np.newaxis]
    
    coord = mds.fit_transform(means_normed)
    stress = mds.stress_
    
    first_coords = coord[0:n_components:]
    second_coords = coord[n_components::]
    
    plt.plot(first_coords.T[0],first_coords.T[1],'bo')
    plt.plot(second_coords.T[0],second_coords.T[1],'yo')
    
    plt.title("stress = " + str(stress) + " sum of the differences = " + str(sum_diff))
    
    titre = 'figure_' + str(n_components) + "_means_comparison_" + legend + ".png"
    plt.savefig(titre)
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
    
#    GMM = GMM3.GaussianMixture(k,covariance_type="full",patience=0,tol=1e-3,reg_covar=1e-6)
#    GMM = VBGMM2.VariationalGaussianMixture(k,tol=1e-3)
    GMM = DPGMM2.VariationalGaussianMixture(k,tol=1e-3)

    #GMM
    for i in range(n_iter):
        print(i)
        print(">>predicting")
#        log_assignements_data,log_assignements_test = GMM.predict_log_assignements(points_data,points_test)
        log_assignements_data = GMM.predict_log_assignements(points_data,draw_graphs=False)
        lower_bound[i] = GMM.convergence_criterion_data[-1]
        print()
            
    plt.title("Lower bounds on " + str(n_iter) + " iterations")
    plt.hist(lower_bound)
    
    directory = GMM.create_path()
    titre = directory + '/repartition_lower_bound_' + str(n_iter) + '_iter.png'
    plt.savefig(titre)
    plt.close("all")
    
    utils.write(directory + 'lower_bounds.csv',lower_bound)