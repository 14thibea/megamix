#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg
from numpy.testing import assert_almost_equal
from megamix.online import GaussianMixture
from megamix.online.base import _log_normal_matrix
from megamix.utils_testing import checking

from scipy.misc import logsumexp
import pytest
import h5py


class TestGaussianMixture_full:
    
    def setup(self):
        self.n_components = 5
        self.dim = 2
        self.n_points = 10
        
        self.file_name = 'test'
        
    def teardown(self):
        checking.remove(self.file_name + '.h5')
        
    def test_initialize(self,window):
        points = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components,window=window)
        GM.initialize(points)
        
        checking.verify_covariance(GM.get('cov'),self.n_components,self.dim)
        checking.verify_means(GM.get('means'),self.n_components,self.dim)
        checking.verify_log_pi(GM.get('log_weights'),self.n_components)
        
        cov_chol = np.empty_like(GM.get('cov'))
        for i in range(self.n_components):
            cov_chol[i] = linalg.cholesky(GM.get('cov')[i],lower=True)
            
        assert_almost_equal(cov_chol,GM.get('cov_chol'))
        assert GM.get('_is_initialized') == True
        
    def test_step_E(self,window):
        points = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components,window=window)
        GM.initialize(points)
        
        log_normal_matrix = _log_normal_matrix(points,GM.get('means'),
                                               GM.get('cov_chol'),'full')
        log_product = log_normal_matrix + GM.get('log_weights')[:,np.newaxis].T
        expected_log_prob_norm = logsumexp(log_product,axis=1)
        expected_log_resp = log_product - expected_log_prob_norm[:,np.newaxis]
        
        predected_log_prob_norm, predected_log_resp = GM._step_E(points)
        
        assert_almost_equal(expected_log_prob_norm,predected_log_prob_norm)
        assert_almost_equal(expected_log_resp,predected_log_resp)
        
    def test_step_M(self,window,update):
        points = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components,window=window,update=update)
        GM.initialize(points)
        
        _,log_resp = GM._step_E(points[:GM.get('window'):])
        GM._sufficient_statistics(points[:GM.get('window'):],log_resp)

        log_weights = np.log(GM.get('N'))
        means = GM.get('X') / GM.get('N')[:,np.newaxis]
        cov = GM.get('S') / GM.get('N')[:,np.newaxis,np.newaxis]
        cov_chol = np.empty_like(cov)
        for i in range(self.n_components):
            cov_chol[i] = linalg.cholesky(cov[i],lower=True)
        
        GM._step_M()
        
        assert_almost_equal(log_weights,GM.get('log_weights'))
        assert_almost_equal(means,GM.get('means'))
        assert_almost_equal(cov,GM.get('cov'))
        assert_almost_equal(cov_chol,GM.get('cov_chol'))
        
    def test_sufficient_statistics(self,window,update):
        points = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components,window=window,update=update)
        GM.initialize(points)
        
        
        _,log_resp = GM._step_E(points[:GM.get('window'):])
        
        points_exp = points[:window:]
        resp = np.exp(log_resp)
        gamma = 1/((GM.get('iter') + window//2)**GM.get('kappa'))
        
        
        # New sufficient statistics
        N = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        N /= window
        
        X = np.dot(resp.T,points_exp)
        X /= window
        

        S = np.zeros((self.n_components,self.dim,self.dim))
        for i in range(self.n_components):
            diff = points_exp - GM.get('means')[i]
            diff_weighted = diff * np.sqrt(resp[:,i:i+1])
            S[i] = np.dot(diff_weighted.T,diff_weighted)
        
        S /= window
        
        # Sufficient statistics update
        expected_N = (1-gamma)*GM.get('N') + gamma*N
        expected_X = (1-gamma)*GM.get('X') + gamma*X
        expected_S = (1-gamma)*GM.get('S') + gamma*S
                     
        expected_S_chol = np.zeros((self.n_components,self.dim,self.dim))
        for i in range(self.n_components):
            expected_S_chol[i] = linalg.cholesky(expected_S[i],lower=True)


        GM._sufficient_statistics(points_exp,log_resp)
        
        assert_almost_equal(expected_N,GM.get('N'))
        assert_almost_equal(expected_X,GM.get('X'))
        assert_almost_equal(expected_S,GM.get('S'))

        
    def test_score(self,window,update):
        points = np.random.randn(self.n_points,self.dim)
        points2 = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components,window=window,update=update)
        
        with pytest.raises(Exception):
            GM.score(points)
        GM.initialize(points)
        GM.fit(points)
        score1 = GM.score(points)
        score2 = GM.score(points2)
        assert score1 > score2

    def test_write_and_read(self,update):
        points = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components,update=update)
        GM.initialize(points)
        
        f = h5py.File(self.file_name + '.h5','w')
        grp = f.create_group('init')
        GM.write(grp)
        f.close()
        
        GM2 = GaussianMixture(self.n_components,update=update)
        
        f = h5py.File(self.file_name + '.h5','r')
        grp = f['init']
        GM2.read_and_init(grp,points)
        f.close()
        
        checking.verify_online_models(GM,GM2)
        
        GM.fit(points)
        GM2.fit(points)
        
        checking.verify_online_models(GM,GM2)
        
    def test_predict_log_resp(self,window,update):
        points = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components,window=window,update=update)
        
        with pytest.raises(Exception):
            GM.predict_log_resp(points)
            
        GM.initialize(points)
        predected_log_resp = GM.predict_log_resp(points)
        _,expected_log_resp = GM._step_E(points)
        
        assert_almost_equal(predected_log_resp,expected_log_resp)
        
    def test_update(self,window):
        points = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components,window=window,update=True)
        
        GM.initialize(points)
        GM.fit(points)
        
        expected_cov_chol = np.zeros((self.n_components,self.dim,self.dim))
        for i in range(self.n_components):
            expected_cov_chol[i] = linalg.cholesky(GM.get('cov')[i],lower=True)
        
        predected_cov_chol = GM.get('cov_chol')
        
        assert_almost_equal(expected_cov_chol,predected_cov_chol)