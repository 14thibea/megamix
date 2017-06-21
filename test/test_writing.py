# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:34:17 2017

:author: Calixi
"""


from megamix import VariationalGaussianMixture
import numpy as np
import time

n_points_init = 1000
n_points = 100
k = 100
dim = 39

points_init = np.random.randn(n_points_init,dim)
points_test = np.random.randn(n_points_init,dim)
points_data = np.random.randn(n_points,dim)
GM = VariationalGaussianMixture(k,init='random')
GM.fit(points_init,points_test)

t = time.time()
GM2 = GM._limiting_model(points_data)
t2 = time.time()

print(t2-t)