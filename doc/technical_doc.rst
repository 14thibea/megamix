Theory of Gaussian Mixture models
=================================

In this part are detailed the equations used in each algorithm.
We use the same notations as Bishop's *Pattern Recognition and Machine Learning* :

* :math:`\{x_1,x_2,...,x_N\}` is the set of points
* :math:`\mu_k` is the center of the kth cluster
* :math:`\pi_k` is the weight of the kth cluster
* :math:`\Sigma_k` is the covariance matrix of the kth cluster

Other notations specific to the methods will be introduced later.

K-means
-------

An iteration of K-means includes:

* The *E step* : a label is assigned to each point (hard assignement) arcording to the means.
* The *M step* : means are computed are computed arcording to the parameters.
* The computation of the *convergence criterion* : the algorithm uses the distortion as described below.

E step
******

..math::
  \left\{
    \begin{split}
    gnagna\\ 
    gnogno
    \end{split}
  \right.

M step
******

Convergence criterion
*********************

Gaussian Mixture Model (GMM)
----------------------------

E step
******

M step
******

Convergence criterion
*********************

Variational Gaussian Mixture Model (VBGMM)
------------------------------------------

E step
******

M step
******

Convergence criterion
*********************

Dirichlet Process Gaussian Mixture Model (DPGMM)
------------------------------------------------

E step
******

M step
******

Convergence criterion
*********************