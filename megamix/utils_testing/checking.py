#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_almost_equal
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
                 
def verify_online_models(GM,GM2):
    assert_almost_equal(GM.get('N'),GM2.get('N'))
    assert_almost_equal(GM.get('log_weights'),GM2.get('log_weights'))
    assert_almost_equal(GM.get('X'),GM2.get('X'))
    assert_almost_equal(GM.get('means'),GM2.get('means'))
    assert GM.get('iter') == GM2.get('iter')
    if GM.get('name') in ['GMM', 'VBGMM']:
        assert_almost_equal(GM.get('S'),GM2.get('S'))
        assert_almost_equal(GM.get('cov'),GM2.get('cov'))
        assert_almost_equal(GM.get('cov_chol'),GM2.get('cov_chol'))
        
        if GM.get('update'):
            assert_almost_equal(GM.get('S_chol'),GM2.get('S_chol'))
            

def verify_batch_models(GM,GM2):
    assert_almost_equal(GM.log_weights,GM2.log_weights)
    assert_almost_equal(GM.means,GM2.means)
    assert GM.iter == GM2.iter
    if GM.name in ['GMM', 'VBGMM']:
        assert_almost_equal(GM.cov,GM2.cov)
