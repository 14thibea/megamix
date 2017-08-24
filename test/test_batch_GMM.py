import numpy as np
from scipy.misc import logsumexp
from numpy.testing import assert_almost_equal
import pytest
import h5py
from megamix.batch import Kmeans, GaussianMixture
from megamix.batch.base import _log_normal_matrix
from megamix.batch.base import _full_covariance_matrices, _spherical_covariance_matrices
from megamix.utils_testing import checking


class TestGaussianMixture:
    
    def setup(self):
        self.n_components = 5
        self.dim = 2
        self.n_points = 10
        
        self.file_name = 'test'
        
    def teardown(self):
        checking.remove(self.file_name + '.h5')
        
        
    def test_initialize(self,type_init,covariance_type):
        points = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components,type_init=type_init,covariance_type=covariance_type)
        GM._initialize(points)
        
        checking.verify_means(GM.means,self.n_components,self.dim)
        checking.verify_log_pi(GM.log_weights,self.n_components)
        assert GM._is_initialized
        
        
    def test_step_E(self,type_init,covariance_type):
        points = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components,type_init=type_init,covariance_type=covariance_type)
        GM._initialize(points)
        
        log_normal_matrix = _log_normal_matrix(points,GM.means,GM.cov,covariance_type)
        log_product = log_normal_matrix + GM.log_weights[:,np.newaxis].T
        expected_log_prob_norm = logsumexp(log_product,axis=1)
        expected_log_resp = log_product - expected_log_prob_norm[:,np.newaxis]
        
        predected_log_prob_norm, predected_log_resp = GM._step_E(points)
        
        assert_almost_equal(expected_log_prob_norm,predected_log_prob_norm)
        assert_almost_equal(expected_log_resp,predected_log_resp)
        
        
    def test_step_M(self,type_init,covariance_type):
        points = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components,type_init=type_init,covariance_type=covariance_type)
        GM._initialize(points)
        
        _,log_resp = GM._step_E(points)
        assignements = np.exp(log_resp)
        
        #Phase 1:
        product = np.dot(assignements.T,points)
        weights = np.sum(assignements,axis=0) + 10 * np.finfo(assignements.dtype).eps
        
        expected_means = product / weights[:,np.newaxis]
        
        #Phase 2:
        if covariance_type=="full":
            expected_cov = _full_covariance_matrices(points,expected_means,weights,assignements,GM.reg_covar,GM.n_jobs)
        elif covariance_type=="spherical":
            expected_cov = _spherical_covariance_matrices(points,expected_means,weights,assignements,GM.reg_covar,GM.n_jobs)
                        
        #Phase 3:
        expected_log_weights = logsumexp(log_resp, axis=0) - np.log(self.n_points)
        
        GM._step_M(points,log_resp)
        
        assert_almost_equal(expected_log_weights,GM.log_weights)
        assert_almost_equal(expected_means,GM.means)
        assert_almost_equal(expected_cov,GM.cov)
        
    
    def test_fit(self,type_init,covariance_type):
        points = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components,type_init=type_init,covariance_type=covariance_type)
        GM._initialize(points)
        GM.init = 'user'
        
        f = h5py.File(self.file_name + '.h5','w')
        grp = f.create_group('init')
        GM.write(grp)
        f.close()
        
        GM2 = GaussianMixture(self.n_components,type_init=type_init,covariance_type=covariance_type)
        
        f = h5py.File(self.file_name + '.h5','r')
        grp = f['init']
        GM2.read_and_init(grp,points)
        f.close()
        
        checking.verify_batch_models(GM,GM2)
        
        GM.fit(points,n_iter_fix=1)
        _,assignements = GM2._step_E(points)
        GM2._step_M(points,assignements)
        GM2.iter += 1
        
        checking.verify_batch_models(GM,GM2)
        
        f = h5py.File(self.file_name + '.h5','r')
        grp = f['init']
        GM.read_and_init(grp,points)
        GM2.read_and_init(grp,points)
        f.close()
        
        GM.fit(points)
        GM2.fit(points)
        
        checking.verify_batch_models(GM,GM2)
        
        
    def test_score(self,type_init,covariance_type):
        points = np.random.randn(self.n_points,self.dim)
        points2 = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components,type_init=type_init,covariance_type=covariance_type)
        
        with pytest.raises(Exception):
            GM.score(points)
        GM._initialize(points)
        GM.fit(points)
        score1 = GM.score(points)
        score2 = GM.score(points2)
        assert score1 > score2
        
    
    def test_write_and_read(self,type_init,covariance_type,):
        points = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components,type_init=type_init,covariance_type=covariance_type)
        GM._initialize(points)
        
        f = h5py.File(self.file_name + '.h5','w')
        grp = f.create_group('init')
        GM.write(grp)
        f.close()
        
        GM2 = GaussianMixture(self.n_components,type_init=type_init,covariance_type=covariance_type)
        
        f = h5py.File(self.file_name + '.h5','r')
        grp = f['init']
        GM2.read_and_init(grp,points)
        f.close()
        
        assert GM.iter == 0
        checking.verify_batch_models(GM,GM2)
        
        _,log_resp = GM._step_E(points)
        _,log_resp2 = GM2._step_E(points)
                
        assert_almost_equal(log_resp,log_resp2)
        
        GM._step_M(points,log_resp)
        GM2._step_M(points,log_resp2)
        
        checking.verify_batch_models(GM,GM2)
        
    def test_predict_log_resp(self,type_init,covariance_type):
        points = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components,type_init=type_init,covariance_type=covariance_type)
        
        with pytest.raises(Exception):
            GM.predict_log_resp(points)
            
        GM._initialize(points)
        predected_log_resp = GM.predict_log_resp(points)
        _,expected_log_resp = GM._step_E(points)
        
        assert_almost_equal(predected_log_resp,expected_log_resp)
        
    def test_write_and_read_KM(self):
        points = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components)
        GM._initialize(points)
        
        f = h5py.File(self.file_name + '.h5','w')
        grp = f.create_group('init')
        GM.write(grp)
        f.close()
        
        predected_KM = Kmeans(self.n_components)
        
        f = h5py.File(self.file_name + '.h5','r')
        grp = f['init']   
        predected_KM.read_and_init(grp,points)
        f.close()
        
        expected_KM = Kmeans(self.n_components)
        
        expected_KM.means = GM.means
        expected_KM.log_weights = GM.log_weights
        expected_KM.iter = GM.iter
        expected_KM._is_initialized = True
                        
        checking.verify_batch_models(predected_KM,expected_KM)
        
    def test_fit_save(self,type_init,covariance_type):
        points = np.random.randn(self.n_points,self.dim)
        GM = GaussianMixture(self.n_components,type_init=type_init,covariance_type=covariance_type)
        
        checking.remove(self.file_name + '.h5')
        GM.fit(points,n_iter_fix=15,saving='log',saving_iter=2,
               file_name=self.file_name)
        f = h5py.File(self.file_name + '.h5','r')
        cpt = 0
        for name in f:
            cpt += 1
            
        assert cpt == 6
        
        checking.remove(self.file_name + '.h5')        
        GM.fit(points,n_iter_fix=15,saving='linear',saving_iter=2,
               file_name=self.file_name)
        f = h5py.File(self.file_name + '.h5','r')
        cpt = 0
        for name in f:
            cpt += 1
            
        assert cpt == 9
        
        checking.remove(self.file_name + '.h5')
        GM.fit(points,n_iter_fix=15,saving='final',saving_iter=2,
               file_name=self.file_name)
        f = h5py.File(self.file_name + '.h5','r')
        cpt = 0
        for name in f:
            cpt += 1
        
        assert cpt == 2    
        