# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 11:51:21 2017

@author: Calixi
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy.random as rnd
import GMM
import CSVreader

if __name__ == '__main__':

    k=4
    list_points = CSVreader.read("D:/Mines/Cours/Stages/Stage_ENS/Problem/EMGaussienne.data")
    list_means = CSVreader.initialization_random(list_points,k)
    list_points = list_points[:,0:-1]
    list_means = list_means[:,0:-1]
    
    
    NUM = 250

    ells = [Ellipse(xy=rnd.rand(2)*10, width=rnd.rand(), height=rnd.rand(), angle=rnd.rand()*360)
            for i in range(NUM)]
    
    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='equal')
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    plt.show()