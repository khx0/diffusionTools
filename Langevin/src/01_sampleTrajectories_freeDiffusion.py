#!/usr/bin/python
# -*- coding: utf-8 -*-
##########################################################################################
# author: Nikolas Schnellbaecher
# contact: khx0@posteo.net
# date: 2018-05-27
# file: 01_sampleTrajectories_freeDiffusion.py
##########################################################################################

import time
import datetime
import sys
import os
import math
import numpy as np

from Langevin import LangevinPropagator_vectorized

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)  

now = datetime.datetime.now()
now = "%s-%s-%s" %(now.year, str(now.month).zfill(2), str(now.day).zfill(2))

BASEDIR = os.path.dirname(os.path.abspath(__file__))
RAWDIR  = os.path.join(BASEDIR, 'raw')
OUTDIR = os.path.join(BASEDIR, 'out')

ensure_dir(RAWDIR)

if __name__ == '__main__':

    sampleTimes = np.array([0.5, 1.0, 3.0, 5.0, 10.0])
    dt = 1.0e-3 # default value in this assay
    
    
    # fix random number seed for reproducibility
    np.random.seed(278123567)
    m = 1000 # numer of independent realizations
    outname = '01_freeDiffusion_m_%d_dt_%.0e' %(m, dt) + '.txt'
    X = LangevinPropagator_vectorized(sampleTimes = sampleTimes, 
                                      m = m)
    assert X.shape == (len(sampleTimes), m), "Error: X shape mismatch."                      
    np.savetxt(os.path.join(RAWDIR, outname), X, fmt = '%.8f')
    
    
    # fix random number seed for reproducibility
    np.random.seed(278123567)
    m = 10000 # numer of independent realizations
    outname = '01_freeDiffusion_m_%d_dt_%.0e' %(m, dt) + '.txt'
    X = LangevinPropagator_vectorized(sampleTimes = sampleTimes, 
                                      m = m)
    assert X.shape == (len(sampleTimes), m), "Error: X shape mismatch."                      
    np.savetxt(os.path.join(RAWDIR, outname), X, fmt = '%.8f')
    
    
    # fix random number seed for reproducibility
    np.random.seed(278123567)
    m = 100000 # numer of independent realizations
    outname = '01_freeDiffusion_m_%d_dt_%.0e' %(m, dt) + '.txt'
    X = LangevinPropagator_vectorized(sampleTimes = sampleTimes, 
                                      m = m)
    assert X.shape == (len(sampleTimes), m), "Error: X shape mismatch."                      
    np.savetxt(os.path.join(RAWDIR, outname), X, fmt = '%.8f')
    
    


