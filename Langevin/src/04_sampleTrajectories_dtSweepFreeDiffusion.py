#!/usr/bin/python
# -*- coding: utf-8 -*-
##########################################################################################
# author: Nikolas Schnellbaecher
# contact: khx0@posteo.net
# date: 2018-05-27
# file: 04_sampleTrajectories_dtSweepFreeDiffusion.py
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
    
    # fix random seed
    np.random.seed(223456789)  # use for release version
    m = 10000
    sampleTimes = np.array([3.0])


    outname = '04_A_freeDiffusion_dt_1e-3_m_%d' %(m) + '.txt'
    X = LangevinPropagator_vectorized(sampleTimes = sampleTimes, 
                                      m = m,
                                      a = 0.0,
                                      sigma = 1.0,
                                      dt = 1.0e-3)
    assert X.shape == (len(sampleTimes), m), "Error: X shape mismatch." 
    np.savetxt(os.path.join(RAWDIR, outname), X, fmt = '%.8f')
    

    outname = '04_B_freeDiffusion_dt_1e-2_m_%d' %(m) + '.txt'
    X = LangevinPropagator_vectorized(sampleTimes = sampleTimes, 
                                      m = m,
                                      a = 0.0,
                                      sigma = 1.0,
                                      dt = 1.0e-2)
    assert X.shape == (len(sampleTimes), m), "Error: X shape mismatch."
    np.savetxt(os.path.join(RAWDIR, outname), X, fmt = '%.8f')

    
    outname = '04_C_freeDiffusion_dt_1e-1_m_%d' %(m) + '.txt'
    X = LangevinPropagator_vectorized(sampleTimes = sampleTimes, 
                                      m = m,
                                      a = 0.0,
                                      sigma = 1.0,
                                      dt = 1.0e-1)
    assert X.shape == (len(sampleTimes), m), "Error: X shape mismatch."    
    np.savetxt(os.path.join(RAWDIR, outname), X, fmt = '%.8f')


    outname = '04_D_freeDiffusion_dt_1e0_m_%d' %(m) + '.txt'
    X = LangevinPropagator_vectorized(sampleTimes = sampleTimes, 
                                      m = m,
                                      a = 0.0,
                                      sigma = 1.0,
                                      dt = 1.0e0)
    assert X.shape == (len(sampleTimes), m), "Error: X shape mismatch."       
    np.savetxt(os.path.join(RAWDIR, outname), X, fmt = '%.8f')
    
    
    
    
    

