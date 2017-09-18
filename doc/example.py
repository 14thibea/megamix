########################
# Prelude to the example
########################
"""
This example is realized with a DP-VBGMM model
The other mixtures and the K-means are working in the same way
The available classes are:
   - Kmeans (kmeans)
   - GaussianMixture (GMM)
   - VariationalGaussianMixture (VBGMM)
   - DPVariationalGaussianMixture (DP-VBGMM)
"""

from megamix.batch import DPVariationalGaussianMixture
import numpy as np

########################
# Features used
########################

"""
Features must be numpy arrays of two dimensions:
the first dimension is the number of points
the second dimension is the dimension of the space
"""

# Here we use a radom set of points for the example
n_points = 10000
dim = 39

points = np.random.randn(n_points,dim)

########################
# Fitting the model
########################

# We choose the number of clusters that we want
n_components = 100

# The model is instantiated
GM = DPVariationalGaussianMixture(n_components)

# The model is fitting
GM.fit(points)

# It is also possible to do early stopping in order to avoid overfitting
points_data = points[:n_points//2:]
points_test = points[n_points//2::]

# In this case the model will fit only on points_data but will use points_test
# to evaluate the convergence criterion.
GM.fit(points_data,points_test)

# Some clusters may disappear with the DP-VBGMM model. You may want to 
# simplify the model by removing the useless information
GM_simple = GM.simplified_model(points)

##########################
# Analysis of the model
##########################

other_points = np.random.randn(n_points,dim)

# We can obtain the log of the reponsibilities of any set of points when the
# model is fitted (or at least initialized)
log_resp = GM.predict_log_resp(other_points)
# log_resp.shape = (n_points,n_components)

# We can obtain the value of the convergence criterion for any set of points
score = GM.score(other_points)

#############################
# Writing or reading a model
#############################

# It is possible to write your model in a group of a h5py file
import h5py

file = h5py.File('DP_VBGMM.h5','w')
grp = file.create_group('model_fitted')

GM.write(grp)
file.close()

# You also can read data from such h5py file to initialize new models
GM_new = DPVariationalGaussianMixture()

file = h5py.File('DP_VBGMM.h5','r')
grp = file['model_fitted']

GM_new.read_and_init(grp,points)
file.close()

# You can also save regurlarly your code while fitting the model by using
# the saving parameter

GM.fit(points,saving='log',directory='mypath',legend='wonderful_model')