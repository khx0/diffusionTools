#!/usr/bin/python
# -*- coding: utf-8 -*-
##########################################################################################
# author: Nikolas Schnellbaecher
# contact: khx0@posteo.net
# date: 2018-05-27
# file: 02_sampleTrajectories_freeDiffusionReflectiveBC.py
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
RAWDIR  = os.path.join(BASEDIR, 'raw')
OUTDIR = os.path.join(BASEDIR, 'out')

ensure_dir(RAWDIR)

def FreeDiffusionReflectiveBC(sampleTimes, m, a = 0.0, sigma = 1.0, dt = 1.0e-3, x0 = 2.0):
    
    nSamples = len(sampleTimes)
    iterations = (sampleTimes / dt).astype(int)        
    iterations[1:] = iterations[1:] - iterations[0:-1]
    totalIterations = (np.cumsum(iterations))[-1]
    
    print("totalIterations = ", totalIterations)
    
    # make sure that the noise amplitude is calculated outside the
    # for loops below to save computation time
    # square roots are numerically quite expensive   
    noise = sigma * np.sqrt(2.0 * dt) # noise amplitude
    
    out = np.zeros((nSamples, m))
    
    # loop over independent realizations
    for j in range(m):
            
        x = x0 # set initial position
        
        for k in range(nSamples):
            
            for i in range(iterations[k]):
            
                x += noise * np.random.normal()
                
                # reflective boundary condition at x = 0.0
                if (x < 0.0):
                    
                    x = -x
                
            out[k, j] = x
     
    return out
    
def FreeDiffusionReflectiveBC_vectorized(sampleTimes, m, a = 0.0, sigma = 1.0, dt = 1.0e-3, x0 = 2.0):
    
    nSamples = len(sampleTimes)
    iterations = (sampleTimes / dt).astype(int)        
    iterations[1:] = iterations[1:] - iterations[0:-1]
    totalIterations = (np.cumsum(iterations))[-1]
        
    # make sure that the noise amplitude is calculated outside the
    # for loops below to save computation time
    # square roots are numerically quite expensive   
    noise = sigma * np.sqrt(2.0 * dt) # noise amplitude
    
    out = np.zeros((nSamples, m))
    
    X = x0 * np.ones((1, m))
    print X.shape
    
    for k in range(nSamples):
            
        for i in range(iterations[k]):
            
            X += noise * np.random.normal(0.0, 1.0, m)
            
            # reflective boundary condition at x = 0.0
            mask = np.argwhere(X < 0.0)
            X[0, mask] = -1.0 * X[0, mask]
                
        out[k, :] = X
     
    return out

if __name__ == '__main__':

    sampleTimes = np.array([0.5, 1.0, 3.0, 5.0, 10.0])
    dt = 1.0e-3
    
    # fix random number seed for reproducibility
    np.random.seed(937162547) # use for release version
    m = 1000
    outname = '02_freeDiffusion_reflectiveBC_m_%d_dt_%.0e' %(m, dt) + '.txt'
    X = FreeDiffusionReflectiveBC_vectorized(sampleTimes = sampleTimes, 
                                  m = m)
    assert X.shape == (len(sampleTimes), m), "Error: X shape mismatch."       
    np.savetxt(os.path.join(RAWDIR, outname), X, fmt = '%.8f')
    
    
    # fix random number seed for reproducibility
    np.random.seed(937162547) # use for release version
    m = 10000
    outname = '02_freeDiffusion_reflectiveBC_m_%d_dt_%.0e' %(m, dt) + '.txt'
    X = FreeDiffusionReflectiveBC_vectorized(sampleTimes = sampleTimes, 
                                  m = m)
    assert X.shape == (len(sampleTimes), m), "Error: X shape mismatch."       
    np.savetxt(os.path.join(RAWDIR, outname), X, fmt = '%.8f')
    
    
    # fix random number seed for reproducibility
    np.random.seed(937162547) # use for release version
    m = 100000
    outname = '02_freeDiffusion_reflectiveBC_m_%d_dt_%.0e' %(m, dt) + '.txt'
    X = FreeDiffusionReflectiveBC_vectorized(sampleTimes = sampleTimes, 
                                  m = m)
    assert X.shape == (len(sampleTimes), m), "Error: X shape mismatch."       
    np.savetxt(os.path.join(RAWDIR, outname), X, fmt = '%.8f')

    
    
    
    
    

