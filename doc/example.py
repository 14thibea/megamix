# This example is realized with a DPGMM model
# The other mixtures and the K-means are working in the same way
# The available classes are:
#   - Kmeans (kmeans)
#   - GaussianMixture (GMM)
#   - VariationalGaussianMixture (VBGMM)
#   - DPVariationalGaussianMixture (DP-VBGMM)


from megamix import DPVariationalGaussianMixture
import numpy as np

########################
# Features used
########################

# Features must be numpy arrays of two dimensions:
# the first dimension is the number of points
# the second dimension is the dimension of the space

# Here we use a radom set of points for the example
n_points = 10000
dim = 39

points = np.random.randn(n_points,dim)

########################
# Fitting the model
########################

# We choose the number of clusters that we want
n_components = 100

# The model is created
GM = DPVariationalGaussianMixture(n_components)


##########################
# Analyse of the model
##########################




#############################
# Writing or reading a model
#############################

