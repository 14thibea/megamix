# -*- coding: utf-8 -*-
"""
Created on Fri May 19 19:10:50 2017

@author: Elina Thibeau-Sutre
"""

"""Setup script for the package"""

import os
from setuptools import setup, find_packages

VERSION = '0.1'
HERE = os.path.dirname(os.path.abspath(__file__))

setup(
    name='megamix',
    version=VERSION,
    packages=find_packages(exclude=['test']),
    include_package_data=True,
    zip_safe=False,
#    install_requires=REQUIREMENTS,

    # metadata for upload to PyPI
    author='Elina Thibeau-Sutre',
    author_email='elina.ts@free.fr',
    description='EM algorithms for unsupervised learning',
    keywords='EM clustering machine learning',
    url='https://github.com/14thibea/Stage_ENS',
    license='MIT',
)