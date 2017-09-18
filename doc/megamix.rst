=============
API Reference
=============

Two versions of the EM algorithms exist : *batch* and *online*:
 * The batch version takes all the point at the same time, using more CPU and memory but may lead to more accurate results.
 * The online version takes the points w by w. In this way the program uses less CPU and memory, but there may be a loss of accuracy.
 
Batch versions of the algorithm
===============================

Four different algorithms have been developped in batch: K-means, GMM, VBGMM and DPGMM

Kmeans
******

.. autoclass:: megamix.batch.Kmeans
    :members:
    :undoc-members:

Gaussian Mixture Model (GMM)
****************************

.. autoclass:: megamix.batch.GaussianMixture
    :members:
    :inherited-members:
    :undoc-members:
	
Variational Gaussian Mixture Model (VBGMM)
******************************************

.. autoclass:: megamix.batch.VariationalGaussianMixture
    :members:
    :inherited-members:
    :undoc-members:

Dirichlet Process Gaussian Mixture Model (DPGMM)
************************************************

.. autoclass:: megamix.batch.DPVariationalGaussianMixture
    :members:
    :inherited-members:
    :undoc-members:

Online versions of the algorithm
================================

Only two algorithms have been developped in batch: K-means and GMM.

Kmeans
******

.. autoclass:: megamix.online.Kmeans
    :members:
    :undoc-members:

Gaussian Mixture Model (GMM)
****************************

.. autoclass:: megamix.online.GaussianMixture
    :members:
    :inherited-members:
    :undoc-members:
