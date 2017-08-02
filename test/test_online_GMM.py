# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg
from numpy.testing import assert_almost_equal
from megamix.online import GaussianMixture
from megamix.online.base import _log_normal_matrix

from scipy.misc import logsumexp
import os

def remove(filename):
    """Remove the file if it exists."""
    if os.path.exists(filename):
        os.remove(filename)
        
def verify_covariance(cov,n_components,dim):
    assert len(cov.shape) == 3
    n_components_cov = cov.shape[0]
    dim_cov1 = cov.shape[1]
    dim_cov2 = cov.shape[2]
    
    assert n_components_cov == n_components
    assert dim_cov1 == dim
    assert dim_cov2 == dim

def verify_means(means,n_components,dim):
    assert len(means.shape) == 2
    n_components_means = means.shape[0]
    dim_means = means.shape[1]
    
    assert n_components_means == n_components
    assert dim_means == dim

def verify_log_pi(log_pi,n_components):
    assert len(log_pi.shape) == 1
    n_components_pi = log_pi.shape[0]
    
    assert n_components_pi == n_components
    assert np.sum(np.exp(log_pi)) - 1.0 < 1e-8 

class TestGaussianMixture_full:
    
    def setup(self):
        self.n_components = 5
        self.dim = 2
        self.n_points = 10
        
    def test_initialize(self,window):
        points = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components,window=window)
        GM.initialize(points)
        
        verify_covariance(GM.cov,self.n_components,self.dim)
        verify_means(GM.means,self.n_components,self.dim)
        verify_log_pi(GM.log_weights,self.n_components)
        
        cov_chol = np.empty_like(GM.cov)
        for i in range(self.n_components):
            cov_chol[i] = linalg.cholesky(GM.cov[i],lower=True)
            
        assert_almost_equal(cov_chol,GM.cov_chol)
        assert GM._is_initialized == True
        
    def test_step_E(self,window):
        points = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components,window=window)
        GM.initialize(points)
        
        log_normal_matrix = _log_normal_matrix(points,GM.means,GM.cov_chol,'full')
        log_product = log_normal_matrix + GM.log_weights[:,np.newaxis].T
        expected_log_prob_norm = logsumexp(log_product,axis=1)
        expected_log_resp = log_product - expected_log_prob_norm[:,np.newaxis]
        
        predected_log_prob_norm, predected_log_resp = GM._step_E(points)
        
        assert_almost_equal(expected_log_prob_norm,predected_log_prob_norm)
        assert_almost_equal(expected_log_resp,predected_log_resp)
        
    def test_step_M(self,window,update):
        points = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components,window=window,update=update)
        GM.initialize(points)
        
        _,log_resp = GM._step_E(points[:GM.window:])
        GM._sufficient_statistics(points[:GM.window:],log_resp)

        log_weights = np.log(GM.N)
        means = GM.X / GM.N[:,np.newaxis]
        cov = GM.S / GM.N[:,np.newaxis,np.newaxis]
        cov_chol = np.empty_like(cov)
        for i in range(self.n_components):
            cov_chol[i] = linalg.cholesky(cov[i],lower=True)
        
        GM._step_M()
        
        assert_almost_equal(log_weights,GM.log_weights)
        assert_almost_equal(means,GM.means)
        assert_almost_equal(cov,GM.cov)
        assert_almost_equal(cov_chol,GM.cov_chol)
        
    def test_sufficient_statistics(self,window):
        points = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components,window=window,update=True)
        GM.initialize(points)
        
        
        _,log_resp = GM._step_E(points[:GM.window:])
        
        points_exp = points[:window:]
        resp = np.exp(log_resp)
        gamma = 1/(((GM.iter + window)//2)**GM.kappa)
        
        
        # New sufficient statistics
        N = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        N /= window
        
        X = np.dot(resp.T,points_exp)
        X /= window
        

        S = np.zeros((self.n_components,self.dim,self.dim))
        for i in range(self.n_components):
            diff = points_exp - GM.means[i]
            diff_weighted = diff * np.sqrt(resp[:,i:i+1])
            S[i] = np.dot(diff_weighted.T,diff_weighted)
#           
        S /= window
        
        # Sufficient statistics update
        expected_N = (1-gamma)*GM.N + gamma*N
        expected_X = (1-gamma)*GM.X + gamma*X
        expected_S = (1-gamma)*GM.S + gamma*S
                     
        expected_S_chol = np.zeros((self.n_components,self.dim,self.dim))
        for i in range(self.n_components):
            expected_S_chol[i] = linalg.cholesky(expected_S[i],lower=True)


        GM._sufficient_statistics(points_exp,log_resp)
        
        assert_almost_equal(expected_N,GM.N)
        assert_almost_equal(expected_X,GM.X)
        assert_almost_equal(expected_S,GM.S)
        assert_almost_equal(expected_S_chol,GM.S_chol)