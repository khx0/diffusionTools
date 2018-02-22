#!/usr/bin/python
# -*- coding: utf-8 -*-
##########################################################################################
# author: Nikolas Schnellbaecher
# contact: khx0@posteo.net
# date: 2018-02-22
# file: StokesEinstein.py
##########################################################################################

import time
import datetime
import sys
import os
import math
import numpy as np

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

now = datetime.datetime.now()
now = "%s-%s-%s" %(now.year, str(now.month).zfill(2), str(now.day).zfill(2))

BASEDIR = os.path.dirname(os.path.abspath(__file__))
RAWDIR = os.path.join(BASEDIR, 'raw')
OUTDIR = os.path.join(BASEDIR, 'out')

"""
    Viscosity table for water:
    first column  -> temperature in degree Celsius
    second column -> viscosity in m Pa * s
"""
viscTable = np.array([[16.0, 1.1081],
                      [17.0, 1.0798],
                      [18.0, 1.0526],
                      [19.0, 1.0266],
                      [20.0, 1.0016],
                      [21.0, 0.9775],
                      [22.0, 0.9544],
                      [23.0, 0.9321],
                      [24.0, 0.9107],
                      [25.0, 0.8900],
                      [26.0, 0.8701],
                      [27.0, 0.8509],
                      [28.0, 0.8324],
                      [29.0, 0.8145],
                      [30.0, 0.7972],
                      [31.0, 0.7805],
                      [32.0, 0.7644],
                      [33.0, 0.7488],
                      [34.0, 0.7337],
                      [35.0, 0.7191],
                      [36.0, 0.7050],
                      [37.0, 0.6913],
                      [38.0, 0.6780],
                      [39.0, 0.6652],
                      [40.0, 0.6527]])

def D_StokesEinstein(temp, eta, radius):
    
    """
    arguments:
        temp = temperature in Kelvin [K]
        eta = viscosity in milli Pascal seconds [m Pa * s]
        radius in nanometer [nm]
    return:
        diffusion coefficient D in units of [(um)^2  / s]
    """
    
    kBValue = 1.38064852
    # 2014 CODATA recommendation from http://physics.nist.gov
    # kB = 1.38064852 x 10^{-23} Joule / Kelvin
    
    # Stokes Einstein relation
    # D = (k_B T) / (6 \pi \eta R)
    return kBValue * temp * 10.0 / (6.0 * np.pi * eta * radius)
    
if __name__ == '__main__':

    pass
    

    

