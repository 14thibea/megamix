.. image:: https://readthedocs.org/projects/megamix/badge/?version=latest
    :target: http://megamix.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
    
.. image:: https://travis-ci.org/14thibea/megamix.svg?branch=master
    :target: https://travis-ci.org/14thibea/megamix
    :alt: Build Status on Travis
   
=======
MeGaMix
=======

.. highlight:: bash

The MeGaMix **python package** provides several **clustering models**
like k-Means and other Gaussian Mixture Models.


Installation
------------

The package depends on *numpy*, *scipy*, *h5py*, *joblib* and *cython* (automatically
installed by the setup script). Install it with::

  $ python setup.py install

Or you can install it with pip::

  $ pip install megamix

Documentation
-------------

See the complete documentation `online <http://megamix.readthedocs.io/en/latest/>`_


Test
----

The package comes with a unit-tests suit. To run it, first install *pytest* on your Python environment::

  $ pip install pytest

Then run the tests with::

  $ pytest