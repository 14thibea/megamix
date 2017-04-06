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

def read(file_name):
    fichier = pd.read_csv(file_name,sep = " ")
    matrix = fichier.as_matrix()
    nb_points = len(matrix)
    dimension = len(matrix[0])
    
    list_points = np.zeros((nb_points,dimension+1))
    list_points[:,:-1] = matrix
    return list_points