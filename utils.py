# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 14:48:20 2017

@author: Calixi
"""

"""
Created on Sun Feb 26 22:02:09 2017
@author: Calixi
"""

import pandas as pd
import numpy as np
from matplotlib.patches import Ellipse
from scipy.special import gammaln

import matplotlib.pyplot as plt

def read(file_name):
    """
    A method to read a file written in CSV style.
    """
    
    fichier = pd.read_csv(file_name,sep = " ")
    matrix = fichier.as_matrix()
    
    return np.asarray(matrix)

def ellipses_multidimensional(cov,mean,d1=0,d2=1):
    """
    A method which creates an object of Ellipse class (from matplotlib.patches).
    As it is only a 2D object dimensions may be precised to project it.
    
    @return: ell (Ellipse)
    """
    
    covariance = np.asarray([[cov[d1,d1],cov[d1,d2]],[cov[d1,d2],cov[d2,d2]]])
    lambda_, v = np.linalg.eig(covariance)
    lambda_sqrt = np.sqrt(lambda_)
    
    width = lambda_sqrt[0] * 2
    height = lambda_sqrt[1] * 2
    
    angle = np.rad2deg(np.arccos(v[0,0]))
    
    ell = Ellipse(xy=(mean[d1], mean[d2]),
              width=width, height=height,
              angle=angle)
    ell.set_facecolor('none')
    ell.set_edgecolor('k')
    return ell

def log_B(W,nu):
    """
    The log of a coefficient involved in the Wishart distribution
    see Bishop book p.693 (B.78)
    """
    
    dim,_ = W.shape
    
    det_W = np.linalg.det(W)
    log_gamma_sum = np.sum(gammaln(np.linspace(nu,nu+dim,num=dim+1)*0.5))
    result = - nu*0.5*np.log(det_W) - nu*dim*0.5*np.log(2)
    result += -dim*(dim-1)*0.25*np.log(np.pi) - log_gamma_sum
    return result

def log_C(alpha):
    """
    The log of a coefficient involved in the Dirichlet distribution
    see Bishop book p.687 (B.23)
    """
    
    return gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))

if __name__ == '__main__':
    
    covariance = np.asarray([[10,0,0],[0,100,-20],[0,-20,10]])
    centre = np.asarray([0,0,0])
    
    result = log_B(covariance,5)
    print(result)
    
    test = log_C(np.asarray([120,150,48,150]))
    print(test)