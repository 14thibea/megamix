Theory of Gaussian Mixture models
=================================

In this part are detailed the equations used in each algorithm.
We use the same notations as Bishop's *Pattern Recognition and Machine Learning*.

Features:

* :math:`\{x_1,x_2,...,x_N\}` is the set of points

Parameters:

* :math:`\mu_k` is the center of the :math:`k^{th}` cluster
* :math:`\pi_k` is the weight of the :math:`k^{th}` cluster
* :math:`\Sigma_k` is the covariance matrix of the :math:`k^{th}` cluster
* :math:`K` is the number of clusters
* :math:`N` is the number of points
* :math:`d` is the dimension of the problem

Other notations specific to the methods will be introduced later.

K-means
-------

An iteration of K-means includes:

* The *E step* : a label is assigned to each point (hard assignement) arcording to the means.
* The *M step* : means are computed according to the parameters.
* The computation of the *convergence criterion* : the algorithm uses the distortion as described below.

E step
******

The algorithm produces a matrix of responsibilities according to the following equation:

.. math::

  r_{nk} = \left\{
    \begin{split}
    & 1 \text{ if } k = \arg\min_{1 \leq j \leq k}\lVert x_n-\mu_j\rVert^2 \\
    & 0 \text{ otherwise}
    \end{split}
  \right.

The value of the case at the :math:`i^{th}` row and :math:`j^{th}` column is 1 if the :math:`i^{th}` point
belongs to the :math:`j^{th}` cluster and 0 otherwise.

M step
******

The mean of a cluster is simply the mean of all the points belonging to this latter:

.. math::

  \mu_{k} = \frac{\sum^N_{n=1}r_{nk}x_n}{\sum^N_{n=1}r_{nk}}
  
The weight of the cluster k can be expressed as:

.. math::

  \pi_{k} = \sum^N_{n=1}r_{nk}

Convergence criterion
*********************

The convergence criterion is the distortion defined as the sum of the norms of the difference between each point
and the mean of the cluster it is belonging to:

.. math::

  D = \sum^N_{n=1}\sum^K_{k=1}r_{nk}\lVert x_n-\mu_k \rVert^2

The distortion should only decrease during the execution of the algorithm. The model stops when the difference between
the value of the convergence criterion at the previous iteration and the current iteration is less or equal to a threshold
:math:`tol` :

.. math::

  D_{previous} - D_{current} \leq tol

Gaussian Mixture Model (GMM)
----------------------------

An iteration of GMM includes:

* The *E step* : :math:`K` probabilities of belonging to each cluster are assigned to each point
* The *M step* : weights, means and covariances are computed  according to the parameters.
* The computation of the *convergence criterion* : the algorithm uses the loglikelihood as described below.

E step
******

The algorithm produces a matrix of responsibilities according to the following equation:

.. math::

  r_{nk} = \frac{\pi_k\mathcal{N}(x_n|\mu_k,\Sigma_k)}{\sum^K_{j=1}\pi_j\mathcal{N}(x_n|\mu_j,\Sigma_j)}

The value of the case at the :math:`i^{th}` row and :math:`j^{th}` column is the probability that the point i belongs to
the cluster j.

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

Pitman-Yor Process Gaussian Mixture Model (PYPGMM)
--------------------------------------------------