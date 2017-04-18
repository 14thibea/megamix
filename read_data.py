# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 09:55:33 2017

@author: Thomas Schatz
"""
import GMM3
import kmeans3
import DPGMM2

from sklearn import mixture
import numpy as np
import pickle

path = 'D:/Mines/Cours/Stages/Stage_ENS/Code/data/data.pickle'
with open(path, 'rb') as fh:
    data = pickle.load(fh)
    
N=1500
    
points = data['BUC']
points = points[:N:]
points = points[:,0:2]
    
covariance_type = "full"

k=10

variance = np.var(points)
#for t in range(10):
#    legend = "MFCC_" + str(t) + "_k" + str(k)
#    means, assignements = kmeans3.k_means(points,k)
#    kmeans3.create_graph(points,means,assignements,legend)
#
#GM = mixture.GaussianMixture(k,covariance_type,tol=5e-5)

DPGMM = DPGMM2.VariationalGaussianMixture(k,"GMM",alpha_0=1.0/k,beta_0=1.0,nu_0=10,patience=0)
print(">>predicting")
log_resp_data,log_resp_test = DPGMM.predict(points,points)
print(">>creating graphs")
legend = str(0) + "MFCC"
DPGMM.create_graph(points,log_resp_data,legend)
DPGMM.create_graph_log(legend)
#for t in range(10):
#means,cov,log_assignements = GMM3.GMM(points,k,draw_graphs=False,initialization="kmeans",covariance_type=covariance_type)
#legend = "MFCC_" + str(0) + "_k" + str(k)
#GMM3.create_graph(points,means,cov,log_assignements,covariance_type,legend)

#GM.fit(points)
#assignements = GM.predict_proba(points)
#_,_,means,_,cov,_= GM._get_parameters()
#prior_prob,means,cov,_= GM._get_parameters()
#log_assignements = GMM3.step_E(points,means,cov,np.log(prior_prob))
#legend = "MFCC_sklearn_k" + str(k)
#GMM3.create_graph(points,means,cov,log_assignements,covariance_type,legend)

#log = GM.score_samples(points)
#print(log)
#print(np.sum(log))
#print(np.mean(log))