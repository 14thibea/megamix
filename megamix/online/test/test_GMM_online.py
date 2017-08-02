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
        
    def teardown(self):
        remove(self.file_name)
        
    def test_initialize_mcw(self):
        points = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components,type_init='mcw')
        GM.initialize(points)
        
        verify_covariance(GM.cov,self.n_components,self.dim)
        verify_means(GM.cov,self.n_components,self.dim)
        verify_log_pi(GM.log_weights,self.n_components)
        assert GM._is_initialized == True
        
    def test_initialize_resp(self):
        points = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components,type_init='resp')
        GM.initialize(points)
        
        verify_covariance(GM.cov,self.n_components,self.dim)
        verify_means(GM.cov,self.n_components,self.dim)
        verify_log_pi(GM.log_weights,self.n_components)
        assert GM._is_initialized == True
        
    def test_step_E(self):
        points = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components)
        GM.initialize(points)
        
        log_normal_matrix = _log_normal_matrix(points,self.means,self.cov,'full')
        log_product = log_normal_matrix + GM.log_weights[:,np.newaxis].T
        expected_log_prob_norm = logsumexp(log_product,axis=1)
        expected_log_resp = log_product - expected_log_prob_norm[:,np.newaxis]
        
        predected_log_prob_norm, predected_log_resp = GM._step_E(points)
        
        assert_almost_equal(expected_log_prob_norm,predected_log_prob_norm)
        assert_almost_equal(expected_log_resp,predected_log_resp)
        
    def test_step_M(self):
        points = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components)
        GM.initialize(points)
        
        log_resp = generate_log_resp