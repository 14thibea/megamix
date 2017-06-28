Theory of Gaussian Mixture models
=================================

In this part are detailed the equations used in each algorithm.
We use the same notations as Bishop's *Pattern Recognition and Machine Learning* :

* :math:`\{x_1,x_2,...,x_N\}` is the set of points
* :math:`\mu_k` is the center of the :math:`k^{th}` cluster
* :math:`\pi_k` is the weight of the :math:`k^{th}` cluster
* :math:`\Sigma_k` is the covariance matrix of the :math:`k^{th}` cluster

Other notations specific to the methods will be introduced later.

K-means
-------

An iteration of K-means includes:

* The *E step* : a label is assigned to each point (hard assignement) arcording to the means.
* The *M step* : means are computed are computed arcording to the parameters.
* The computation of the *convergence criterion* : the algorithm uses the distortion as described below.

E step
******

The algorithm produces a matrix of responsibilities according to the following equation:

.. math::

  r_{nk} = \left\{
    \begin{split}
    & 1 \text{ if } k = \arg\min_{1<=j<=k}||x_i-c_j||_2^2 \\
    & 0 \text{ otherwise}
    \end{split}
  \right.

The value of the case at the :math:`i^{th}` row and :math:`j^{th}` column is 1 if the :math:`i^{th}` point belong to the :math:`j^{th}` cluster and 0 otherwise

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