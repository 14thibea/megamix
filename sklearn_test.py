# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:33:23 2017

@author: Calixi
"""

from sklearn import mixture
import CSVreader

list_points = CSVreader.read("D:/Mines/Cours/Stages/Stage_ENS/Problem/EMGaussienne.data")
list_points = list_points[:,0:-1]

list_points2 = CSVreader.read("D:/Mines/Cours/Stages/Stage_ENS/Problem/EMGaussienne.test")
list_points2 = list_points2[:,0:-1]

GM = mixture.GaussianMixture(n_components=3,tol=0.00001,init_params='random')
array = GM.fit(list_points)
print(array)
array2 = GM.predict(list_points2)
print(array2)