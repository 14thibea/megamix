# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:34:17 2017

:author: Calixi
"""


from megamix import DPVariationalGaussianMixture
import numpy as np

n_points = 10000
k = 100
dim = 39

points = np.random.randn(n_points,dim)
GM = DPVariationalGaussianMixture(k,init='kmeans')
GM._initialize(points)