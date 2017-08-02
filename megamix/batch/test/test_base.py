import numpy as np
from scipy import linalg
from numpy.testing import assert_almost_equal
from megamix.batch.base import _log_normal_matrix, _compute_precisions_chol
from megamix.batch.base import _full_covariance_matrices, _spherical_covariance_matrices

def generate_mixing_coefficients(n_components):
    pi = np.abs(np.random.randn(n_components))
    return pi/pi.sum()

def generate_covariance_matrices_full(n_components,n_features):
    cov = np.empty((n_components,n_features,n_features))
    for i in range(n_components):
        X = np.random.randn(10*n_features,n_features)
        cov[i] = np.dot(X.T,X)
        
    return cov

def generate_resp(n_points,n_components):
    resp = np.abs(np.random.randn(n_points,n_components))
    return resp/resp.sum(axis=1)[:,np.newaxis]


def test_log_normal_matrix_full():
    n_points, n_components, n_features = 10,5,2
    
    points = np.random.randn(n_points,n_features)
    means = np.random.randn(n_components,n_features)
    cov = generate_covariance_matrices_full(n_components,n_features)
    
    # Beginnig of the test
    log_det_cov = np.log(np.linalg.det(cov))
    precisions = np.linalg.inv(cov)
    log_prob = np.empty((n_points,n_components))
    for i in range(n_components):
        diff = points - means[i]
        y = np.dot(diff,np.dot(precisions[i],diff.T))
        log_prob[:,i] = np.diagonal(y)
        
    expected_log_normal_matrix = -0.5 * (n_features * np.log(2*np.pi) +
                                         log_prob + log_det_cov)
    
    predected_log_normal_matrix = _log_normal_matrix(points,means,cov,'full')
    
    assert_almost_equal(expected_log_normal_matrix,predected_log_normal_matrix)
    
def test_compute_precisions_chol_full():
    n_components, n_features = 5,2
    
    cov = generate_covariance_matrices_full(n_components,n_features)
    expected_precisions_chol = np.empty((n_components,n_features,n_features))
    for i in range(n_components):
        cov_chol = linalg.cholesky(cov[i],lower=True)
        expected_precisions_chol[i] = np.linalg.inv(cov_chol).T
        
    predected_precisions_chol = _compute_precisions_chol(cov,'full')
    
    assert_almost_equal(expected_precisions_chol,predected_precisions_chol)
    
def test_full_covariance_matrices():
    n_points, n_components, n_features = 10,5,2
    
    points = np.random.randn(n_points,n_features)
    means = np.random.randn(n_components,n_features)
    pi = generate_mixing_coefficients(n_components)
    resp = generate_resp(n_points,n_components)
    weights = pi * n_points
    reg_covar = 1e-6
    
    expected_full_covariance_matrices = np.empty((n_components,n_features,n_features))
    for i in range(n_components):
        diff = points - means[i]
        diff_weighted = diff*resp[:,i:i+1]
        cov = 1/weights[i] * np.dot(diff_weighted.T,diff)
        cov.flat[::n_features+1] += reg_covar
        expected_full_covariance_matrices[i] = cov
                                         
    predected_full_covariance_matrices = _full_covariance_matrices(points,means,weights,resp,reg_covar)
        
    assert_almost_equal(expected_full_covariance_matrices,predected_full_covariance_matrices)
    
def test_spherical_covariance_matrices():
    n_points, n_components, n_features = 10,5,2
    
    points = np.random.randn(n_points,n_features)
    means = np.random.randn(n_components,n_features)
    pi = generate_mixing_coefficients(n_components)
    resp = generate_resp(n_points,n_components)
    weights = pi * n_points
    reg_covar = 1e-6
    
    expected_full_covariance_matrices = np.empty(n_components)
    for i in range(n_components):
        diff = points - means[i]
        diff_weighted = diff * resp[:,i:i+1]
        product = diff * diff_weighted
        expected_full_covariance_matrices[i] = np.sum(product)/weights[i] + reg_covar    
    expected_full_covariance_matrices /= n_features
                                   
    predected_full_covariance_matrices = _spherical_covariance_matrices(points,means,weights,resp,reg_covar)
        
    assert_almost_equal(expected_full_covariance_matrices,predected_full_covariance_matrices)