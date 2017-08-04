#!/usr/bin/env python
# -*- coding: utf-8 -*-
def pytest_addoption(parser):
    parser.addoption("--online", action="store_true",
        help="run only tests for online (window=1)")

def pytest_generate_tests(metafunc):
    if 'window' in metafunc.fixturenames:
        if metafunc.config.getoption('online'):
            list_window=[1]
        else:
            list_window=[1,2,5]
        metafunc.parametrize("window", list_window)
        
    if 'update' in metafunc.fixturenames:
        metafunc.parametrize("update", [True,False])
        
    if 'type_init' in metafunc.fixturenames:
        metafunc.parametrize("type_init", ['resp','mcw'])
        
    if 'covariance_type' in metafunc.fixturenames:
        metafunc.parametrize("covariance_type", ['full','spherical'])