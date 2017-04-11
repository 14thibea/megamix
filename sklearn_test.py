# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:33:23 2017

@author: Calixi
"""

from sklearn import mixture
import utils
import matplotlib.pyplot as plt
import os
import GMM3
import numpy as np

def draw_graph(k,points,means,cov,resp,t):
    
    n_points = len(points)
    couleurs = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        
    log_like = GMM3.log_likelihood(points,means,cov,np.log(resp))
    
    fig = plt.figure()
    plt.title("log likelihood = " + str(log_like))
    ax = fig.add_subplot(111)
        
        
    for j in range(k):
        for i in range(n_points):        
            ax.plot(points[i][0],points[i][1],couleurs[j]+'o',alpha = resp[i][j]/5)
        ax.plot(means[j][0],means[j][1],'kx')
        ell = utils.ellipses(cov[j],means[j])
        ax.add_artist(ell)
        print("cluster " + str(j) + " finished")
    print()
            
    dir_path = 'VBGMM/sklearn/'
    directory = os.path.dirname(dir_path)
    
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)  
        
    titre = directory + '/figure_' + str(t)
                
    plt.savefig(titre)
    plt.close("all")
        
    

if __name__ == '__main__':
    
    points = utils.read("D:/Mines/Cours/Stages/Stage_ENS/Code/Problem/data/EMGaussienne.data")
    points2 = utils.read("D:/Mines/Cours/Stages/Stage_ENS/Code/Problem/data/EMGaussienne.test")
    
    k=4
    
    GM = mixture.BayesianGaussianMixture(k)
    for t in range(20):       
        GM.fit(points)
        assignements = GM.predict_proba(points)
        _,_,means,_,cov,_= GM._get_parameters()
#        _,means,cov,_= GM._get_parameters()
    
        draw_graph(k,points,means,cov,assignements,t)
#        GMM3.create_graph(points,means,cov,np.log(assignements),"full",t)
    
    