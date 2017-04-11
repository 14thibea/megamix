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

def read(file_name):
    fichier = pd.read_csv(file_name,sep = " ")
    matrix = fichier.as_matrix()
    
    return np.asarray(matrix)

def ellipses(covariance,mean):
      
    lambda_, v = np.linalg.eig(covariance)
    lambda_ = np.sqrt(lambda_)
    ell = Ellipse(xy=(mean[0], mean[1]),
              width=lambda_[0]*2, height=lambda_[1]*2,
              angle=np.rad2deg(np.arccos(v[0, 0])))
    ell.set_facecolor('none')
    ell.set_edgecolor('k')
    return ell