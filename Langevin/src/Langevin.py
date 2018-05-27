#!/usr/bin/python
# -*- coding: utf-8 -*-
##########################################################################################
# author: Nikolas Schnellbaecher
# contact: khx0@posteo.net
# date: 2018-05-27
# file: Langevin.py
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
ensure_dir(OUTDIR)

def LangevinPropagator(sampleTimes, m, a = 0.0, sigma = 1.0, dt = 1.0e-3, x0 = 2.0):
    
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
        
            out[k, j] = x
     
    return out

def LangevinPropagator_vectorized(sampleTimes, m, a = 0.0, sigma = 1.0, dt = 1.0e-3, x0 = 2.0):
    '''
    Vectorized version of the LangevinPropagator function, making use of
    numpy's vectorization possibilities.
    '''
    nSamples = len(sampleTimes)
    iterations = (sampleTimes / dt).astype(int)        
    iterations[1:] = iterations[1:] - iterations[0:-1]
    totalIterations = (np.cumsum(iterations))[-1]
        
    # make sure that the noise amplitude is calculated outside the
    # for loops below to save computation time
    # square roots are numerically quite expensive   
    noise = sigma * np.sqrt(2.0 * dt) # noise amplitude
    drift = a * dt
    
    out = np.zeros((nSamples, m))
    
    X = x0 * np.ones((1, m)) # row element
    
    for k in range(nSamples):
        
        for i in range(iterations[k]):
        
            X += drift * X + noise * np.random.normal(0.0, 1.0, m)
        
        out[k, :] = X
    
    return out

def getMoments(X, sampleTimes):
    ######################################################################################
    # FIRST MOMENT AND MSD
    # evaluate the first moments and the MSD from the given trajectories
    firstMoments = np.mean(X, axis = 1)
    timePoints = len(sampleTimes)
    MSD = np.zeros((timePoints,))
    for i in range(timePoints):
        MSD[i] = np.mean(np.square((X[i, : ] - firstMoments[i])))
    ######################################################################################
    return firstMoments, MSD
    
if __name__ == '__main__':
    
    pass
    
    


