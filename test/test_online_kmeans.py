#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_almost_equal
from megamix.online import Kmeans, GaussianMixture
from megamix.online import dist_matrix
from megamix.utils_testing import checking
from scipy import linalg

import pytest
import h5py

def test_dist_matrix():
    n_points,n_components,dim = 10,5,2
    points = np.random.randn(n_points,dim)
    means = np.random.randn(n_components,dim)
    
    expected_dist_matrix = np.zeros((n_points,n_components))
    for i in range(n_points):
        for j in range(n_components):
            expected_dist_matrix[i,j] = np.linalg.norm(points[i] - means[j])
    
    predected_dist_matrix = dist_matrix(points,means)
    
    assert_almost_equal(expected_dist_matrix,predected_dist_matrix)

class TestKmeans:
    
    def setup(self):
        self.n_components = 5
        self.dim = 2
        self.n_points = 10
        
        self.file_name = 'test.h5'
        
    def teardown(self):
        checking.remove(self.file_name)
        
    def test_initialize(self,window):
        points = np.random.randn(self.n_points,self.dim)
        KM = Kmeans(self.n_components,window=window)
        KM.initialize(points)
        
        checking.verify_means(KM.get('means'),self.n_components,self.dim)
        checking.verify_log_pi(np.log(KM.get('N')),self.n_components)
        assert KM.get('_is_initialized')
        
    def test_step_E(self,window):
        points = np.random.randn(self.n_points,self.dim)
        KM = Kmeans(self.n_components,window=window)
        KM.initialize(points)
        
        expected_assignements = np.zeros((self.n_points,self.n_components))
        M = dist_matrix(points,KM.get('means'))
        for i in range(self.n_points):
            index_min = np.argmin(M[i]) #the cluster number of the ith point is index_min
            if (isinstance(index_min,np.int64)):
                expected_assignements[i][index_min] = 1
            else: #Happens when two points are equally distant from a cluster mean
                expected_assignements[i][index_min[0]] = 1
                
        predected_assignements = KM._step_E(points)
        
        assert_almost_equal(expected_assignements,predected_assignements)
        
    def test_step_M(self,window):
        points = np.random.randn(self.n_points,self.dim)
        KM = Kmeans(self.n_components,window=window)
        KM.initialize(points)
        points_window = points[:window:]
        assignements = KM._step_E(points_window)
        
        N = assignements.sum(axis=0) / window
        X = np.dot(assignements.T,points_window) / window
                  
        gamma = 1/((KM.get('iter') + window//2)**KM.get('kappa'))
        
        expected_N = (1-gamma)*KM.get('N') + gamma*N
        expected_X = (1-gamma)*KM.get('X') + gamma*X
        expected_means = expected_X / expected_N[:,np.newaxis]
        
        KM._step_M(points_window,assignements)
                     
        assert_almost_equal(expected_N,KM.get('N'))
        assert_almost_equal(expected_X,KM.get('X'))
        assert_almost_equal(expected_means,KM.get('means'))
        
        
    def test_score(self,window):
        points = np.random.randn(self.n_points,self.dim)
        KM = Kmeans(self.n_components)
        
        with pytest.raises(Exception):
            KM.score(points)
        KM.initialize(points)
        KM.fit(points)
        score1 = KM.score(points)
        score2 = KM.score(points+2)
        assert score1 < score2
        
    
    def test_predict_assignements(self,window):
        points = np.random.randn(self.n_points,self.dim)
        KM = Kmeans(self.n_components)
        
        with pytest.raises(Exception):
            KM.score(points)
        KM.initialize(points)
        
        expected_assignements = KM._step_E(points)
        
        predected_assignements = KM.predict_assignements(points)
        
        assert_almost_equal(expected_assignements,predected_assignements)
        
        
    def test_write_and_read(self):
        points = np.random.randn(self.n_points,self.dim)
        KM = Kmeans(self.n_components)
        KM.initialize(points)
        
        f = h5py.File(self.file_name,'w')
        grp = f.create_group('init')
        KM.write(grp)
        f.close()
        
        KM2 = Kmeans(self.n_components)
        
        f = h5py.File(self.file_name,'r')
        grp = f['init']
        KM2.read_and_init(grp,points)
        f.close()
        
        checking.verify_online_models(KM,KM2)
        
        KM.fit(points)
        KM2.fit(points)
        
        checking.verify_online_models(KM,KM2)
        
    def test_write_and_read_GM(self,update):
        points = np.random.randn(self.n_points,self.dim)
        KM = Kmeans(self.n_components)
        KM.initialize(points)
        
        f = h5py.File(self.file_name,'w')
        grp = f.create_group('init')
        KM.write(grp)
        f.close()
        
        predected_GM = GaussianMixture(self.n_components,update=update)
        
        f = h5py.File(self.file_name,'r')
        grp = f['init']   
        with pytest.warns(UserWarning):
            predected_GM.read_and_init(grp,points)
        f.close()
        
        expected_GM = GaussianMixture(self.n_components,update=update)
        
        expected_GM.set('means',KM.get('means'))
        expected_GM._initialize_cov(points)
        
        # Computation of self.cov_chol
        expected_GM.set('cov_chol',np.empty(expected_GM.get('cov').shape))
        for i in range(self.n_components):
            expected_GM.cov_chol[i] = linalg.cholesky(expected_GM.get('cov')[i],lower=True)
                        
        expected_GM._initialize_weights(points)
        expected_GM.set('iter',KM.get('iter'))

        weights = np.exp(expected_GM.log_weights)
        expected_GM.set('N', weights)
        expected_GM.set('X', expected_GM.get('means') * expected_GM.get('N')[:,np.newaxis])
        expected_GM.set('S', expected_GM.get('cov') * expected_GM.get('N')[:,np.newaxis,np.newaxis])
                        
        checking.verify_online_models(predected_GM,expected_GM)

    def test_fit_save(self,window):
        points = np.random.randn(self.n_points,self.dim)
        KM = Kmeans(self.n_components,window=window)
        
        checking.remove(self.file_name + '.h5')
        KM.initialize(points)
        KM.fit(points,saving='linear',saving_iter=2,
               file_name=self.file_name)
        f = h5py.File(self.file_name + '.h5','r')
        cpt = 0
        for name in f:
            cpt += 1
            
        assert cpt == self.n_points//(2*window)
        
        checking.remove(self.file_name + '.h5')        
        KM.fit(points,saving='log',saving_iter=2,
               file_name=self.file_name)
        f = h5py.File(self.file_name + '.h5','r')
        cpt = 0
        for name in f:
            cpt += 1
            
        assert cpt == 1 + int(np.log(self.n_points/window)/np.log(2))