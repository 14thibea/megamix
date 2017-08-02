import numpy as np

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
