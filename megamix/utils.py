# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 14:48:20 2017

@author: Elina Thibeau-Sutre
"""

import pandas as pd
import numpy as np

def read(file_name):
    """
    A method to read a file written in CSV style.
    """
    
    fichier = pd.read_csv(file_name,sep = " ")
    matrix = fichier.as_matrix()
    
    return np.asarray(matrix)

def write(file_name,data,sep=' '):
    """
    A method used to write a CSV file from data
    """
    df = pd.DataFrame(data)
    df.to_csv(file_name,sep=sep,header=False,index=False)