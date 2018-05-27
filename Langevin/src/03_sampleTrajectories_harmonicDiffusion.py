#!/usr/bin/python
# -*- coding: utf-8 -*-
##########################################################################################
# author: Nikolas Schnellbaecher
# contact: khx0@posteo.net
# date: 2018-05-27
# file: 03_sampleTrajectories_harmonicDiffusion.py
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
    dt = 1.0e-3
    
    # fix random seed
    np.random.seed(9822165) # use for release version
    m = 1000
    outname = '03_harmonicDiffusion_m_%d_dt_%.0e' %(m, dt) + '.txt'
    X = LangevinPropagator_vectorized(sampleTimes = sampleTimes, 
                                      m = m,
                                      a = -1.0)                
    assert X.shape == (len(sampleTimes), m), "Error: X shape mismatch."   
    np.savetxt(os.path.join(RAWDIR, outname), X, fmt = '%.8f')
    
    
    # fix random seed
    np.random.seed(9822165) # use for release version
    m = 10000
    outname = '03_harmonicDiffusion_m_%d_dt_%.0e' %(m, dt)  + '.txt'
    X = LangevinPropagator_vectorized(sampleTimes = sampleTimes, 
                                      m = m,
                                      a = -1.0)                
    assert X.shape == (len(sampleTimes), m), "Error: X shape mismatch."   
    np.savetxt(os.path.join(RAWDIR, outname), X, fmt = '%.8f')
    
    
    # fix random seed
    np.random.seed(9822165) # use for release version
    m = 100000
    outname = '03_harmonicDiffusion_m_%d_dt_%.0e' %(m, dt)  + '.txt'
    X = LangevinPropagator_vectorized(sampleTimes = sampleTimes, 
                                      m = m,
                                      a = -1.0)                
    assert X.shape == (len(sampleTimes), m), "Error: X shape mismatch."   
    np.savetxt(os.path.join(RAWDIR, outname), X, fmt = '%.8f')
    

