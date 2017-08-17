import numpy as np
from scipy import linalg
from numpy.testing import assert_almost_equal
import pytest
import h5py
from megamix.batch import Kmeans,GaussianMixture,dist_matrix
from megamix.utils_testing import checking

def test_dist_matrix():
    n_points,n_components,dim = 10,5,2
    points = np.random.randn(n_points,dim)
    means = np.random.randn(n_components,dim)
    
    expected_dist_matrix = np.zeros((n_points,n_components))
    for i in range(n_points):
        for j in range(n_components):
            expected_dist_matrix[i,j] = np.linalg.norm(points[i] - means[j])
    
    predected_dist_matrix = dist_matrix(points,means)
    
    assert_almost_equal(expected_dist_matrix,predected_dist_matrix,9)

class TestKmeans:
    
    def setup(self):
        self.n_components = 5
        self.dim = 2
        self.n_points = 10
        
        self.file_name = 'test'
        
    def teardown(self):
        checking.remove(self.file_name + 'h5')
        
        
    def test_initialize(self):
        points = np.random.randn(self.n_points,self.dim)
        KM = Kmeans(self.n_components)
        KM._initialize(points)
        
        checking.verify_means(KM.means,self.n_components,self.dim)
        checking.verify_log_pi(KM.log_weights,self.n_components)
        assert KM._is_initialized
        
        
    def test_step_E(self):
        points = np.random.randn(self.n_points,self.dim)
        KM = Kmeans(self.n_components)
        KM._initialize(points)
        
        expected_assignements = np.zeros((self.n_points,self.n_components))
        M = dist_matrix(points,KM.means)
        for i in range(self.n_points):
            index_min = np.argmin(M[i]) #the cluster number of the ith point is index_min
            if (isinstance(index_min,np.int64)):
                expected_assignements[i][index_min] = 1
            else: #Happens when two points are equally distant from a cluster mean
                expected_assignements[i][index_min[0]] = 1
                
        predected_assignements = KM._step_E(points)
        
        assert_almost_equal(expected_assignements,predected_assignements)
        
    def test_step_M(self):
        points = np.random.randn(self.n_points,self.dim)
        KM = Kmeans(self.n_components)
        KM._initialize(points)
        assignements = KM._step_E(points)
        
        expected_means = KM.means.copy()
        for i in range(self.n_components):
            assignements_i = assignements[:,i:i+1]
            n_set = np.sum(assignements_i)
            idx_set,_ = np.where(assignements_i==1)
            sets = points[idx_set]
            if n_set > 0:
                expected_means[i] = np.asarray(np.sum(sets, axis=0)/n_set)

        
        KM._step_M(points,assignements)
                     
        assert_almost_equal(expected_means,KM.means)
        
        
    def test_score(self,window):
        points = np.random.randn(self.n_points,self.dim)
        KM = Kmeans(self.n_components)
        
        with pytest.raises(Exception):
            KM.score(points)
        KM.fit(points)
        score1 = KM.score(points)
        score2 = KM.score(points+2)
        assert score1 < score2
        
    
    def test_predict_assignements(self,window):
        points = np.random.randn(self.n_points,self.dim)
        KM = Kmeans(self.n_components)
        
        with pytest.raises(Exception):
            KM.distortion(points)
        KM._initialize(points)
        
        expected_assignements = KM._step_E(points)
        
        predected_assignements = KM.predict_assignements(points)
        
        assert_almost_equal(expected_assignements,predected_assignements)
        
        
    def test_write_and_read(self):
        points = np.random.randn(self.n_points,self.dim)
        KM = Kmeans(self.n_components)
        KM._initialize(points)
        
        f = h5py.File(self.file_name + '.h5','w')
        grp = f.create_group('init')
        KM.write(grp)
        f.close()
        
        KM2 = Kmeans(self.n_components)
        
        f = h5py.File(self.file_name + '.h5','r')
        grp = f['init']
        KM2.read_and_init(grp,points)
        f.close()
        
        checking.verify_batch_models(KM,KM2)
        
        assignements = KM._step_E(points)
        assignements2 = KM2._step_E(points)
        
        assert_almost_equal(assignements,assignements2)
        
        KM._step_M(points,assignements)
        KM2._step_M(points,assignements2)
        
        checking.verify_batch_models(KM,KM2)
        
        
    def test_fit(self):
        points = np.random.randn(self.n_points,self.dim)
        KM = Kmeans(self.n_components)
        KM._initialize(points)
        KM.init = 'user'
        
#        checking.remove(self.file_name + '.h5')
        f = h5py.File(self.file_name + '.h5','w')
        grp = f.create_group('init')
        KM.write(grp)
        f.close()
        
        KM2 = Kmeans(self.n_components)
        
        f = h5py.File(self.file_name + '.h5','r')
        grp = f['init']
        KM2.read_and_init(grp,points)
        f.close()
        
        checking.verify_batch_models(KM,KM2)
        
        KM.fit(points,n_iter_fix=1)
        assignements = KM2._step_E(points)
        KM2._step_M(points,assignements)
        KM2.iter += 1
        
        checking.verify_batch_models(KM,KM2)
        
        f = h5py.File(self.file_name + '.h5','r')
        grp = f['init']
        KM.read_and_init(grp,points)
        KM2.read_and_init(grp,points)
        f.close()
        
        KM.fit(points)
        KM2.fit(points)
        
        checking.verify_batch_models(KM,KM2)

        
    def test_write_and_read_GM(self):
        points = np.random.randn(self.n_points,self.dim)
        KM = Kmeans(self.n_components)
        KM._initialize(points)
        
        f = h5py.File(self.file_name + '.h5','w')
        grp = f.create_group('init')
        KM.write(grp)
        f.close()
        
        predected_GM = GaussianMixture(self.n_components)
        
        f = h5py.File(self.file_name + '.h5','r')
        grp = f['init']   
        with pytest.warns(UserWarning):
            predected_GM.read_and_init(grp,points)
        f.close()
        
        expected_GM = GaussianMixture(self.n_components)
        
        expected_GM.means = KM.means
        expected_GM._initialize_cov(points)
        expected_GM.log_weights = KM.log_weights
        expected_GM.iter = KM.iter
        expected_GM._is_initialized = True
                        
        checking.verify_batch_models(predected_GM,expected_GM)
        
        
    def test_fit_save(self):
        points = np.random.randn(self.n_points,self.dim)
        KM = Kmeans(self.n_components)
        
        checking.remove(self.file_name + '.h5')
        KM.fit(points,n_iter_fix=15,saving='log',saving_iter=2,
               file_name=self.file_name)
        f = h5py.File(self.file_name + '.h5','r')
        cpt = 0
        for name in f:
            cpt += 1
            
        assert cpt == 6
        
        checking.remove(self.file_name + '.h5')        
        KM.fit(points,n_iter_fix=15,saving='linear',saving_iter=2,
               file_name=self.file_name)
        f = h5py.File(self.file_name + '.h5','r')
        cpt = 0
        for name in f:
            cpt += 1
            
        assert cpt == 9
        
        checking.remove(self.file_name + '.h5')
        KM.fit(points,n_iter_fix=15,saving='final',saving_iter=2,
               file_name=self.file_name)
        f = h5py.File(self.file_name + '.h5','r')
        cpt = 0
        for name in f:
            cpt += 1
        
        assert cpt == 2