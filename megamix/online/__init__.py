# -*- coding: utf-8 -*-
"""
Created on Fri May 19 18:50:46 2017

@author: Elina Thibeau-Sutre
"""
from .VBGMM import VariationalGaussianMixture
from .GMM import GaussianMixture
from .kmeans import Kmeans

__all__ = ['VariationalGaussianMixture',
           'GaussianMixture','Kmeans']