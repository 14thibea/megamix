# -*- coding: utf-8 -*-
#
#Created on Fri May 19 19:10:50 2017
#
#author: Elina Thibeau-Sutre
#

"""Setup script for the package"""

import os
from setuptools import setup, find_packages
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

VERSION = '0.3.2'
HERE = os.path.dirname(os.path.abspath(__file__))

ON_RTD = os.environ.get('READTHEDOCS', None) == 'True'

ext_modules = [] if ON_RTD else [

    Extension(
        "megamix.online.cython_version.base_cython",
        ["megamix/online/cython_version/base_cython.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()],
    ),

    Extension(
        "megamix.online.cython_version.basic_operations",
        ["megamix/online/cython_version/basic_operations.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()],
    ),
            
    Extension(
        "megamix.online.cython_version.GMM_cython",
        ["megamix/online/cython_version/GMM_cython.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()],
    ),
            
    Extension(
        "megamix.online.cython_version.kmeans_cython",
        ["megamix/online/cython_version/kmeans_cython.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()],
    ),

]

REQUIREMENTS = [] if ON_RTD else [
    'numpy >= 1.11.3',
    'h5py >= 2.6.0',
    'scipy >= 0.18.1',
    'joblib >= 0.11'
	'cython'
]

setup(
    name='megamix',
    version=VERSION,
    packages=find_packages(exclude=['test']),
                          
    # Comment this if you cannot compile
    ext_modules=cythonize(ext_modules),
                         
    include_package_data=True,
    zip_safe=False,
    install_requires=REQUIREMENTS,

    # metadata for upload to PyPI
    author='Elina Thibeau-Sutre',
    author_email='elina.ts@free.fr',
    description='EM algorithms for unsupervised learning',
    keywords='EM clustering machine learning',
    url='https://github.com/14thibea/Stage_ENS',
    license='MIT',
)

#try:
#    setup(ext_modules=cythonize(ext_modules))
#except:
#    print("It didn't compile")