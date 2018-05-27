#!/usr/bin/python
# -*- coding: utf-8 -*-
##########################################################################################
# author: Nikolas Schnellbaecher
# contact: khx0@posteo.net
# date: 2018-05-27
# file: 04_analyze_dtSweepFreeDiffusion.py
##########################################################################################    

import time
import datetime
import sys
import os
import math
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.pyplot import legend
import matplotlib.colors as colors
import matplotlib.cm as cmx
# from scipy import stats
# import scipy

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)  

now = datetime.datetime.now()
now = "%s-%s-%s" %(now.year, str(now.month).zfill(2), str(now.day).zfill(2))

BASEDIR = os.path.dirname(os.path.abspath(__file__))
RAWDIR  = os.path.join(BASEDIR, 'raw')
OUTDIR = os.path.join(BASEDIR, 'out')

ensure_dir(OUTDIR)

def getFigureProps(width, height, lFrac = 0.17, rFrac = 0.9, bFrac = 0.17, tFrac = 0.9):
    '''
    True size scaling auxiliary function to setup mpl plots with a desired size in cm.
    Specify widht and height in cm.
    lFrac = left fraction   in [0, 1]
    rFrac = right fraction  in [0, 1]
    bFrac = bottom fraction in [0, 1]
    tFrac = top fraction    in [0, 1]
    returns:
        fWidth = figure width
        fHeight = figure height
    These figure width and height values can then be used to create a figure instance 
    of the desired size, such that the actual plotting canvas has the specified
    target width and height, as provided by the input parameters of this function.
    '''
    axesWidth = width / 2.54    # convert to inches (1 inch = 2.54 cm)
    axesHeight = height / 2.54  # convert to inches
    fWidth = axesWidth / (rFrac - lFrac)
    fHeight = axesHeight / (tFrac - bFrac)
    return fWidth, fHeight, lFrac, rFrac, bFrac, tFrac

def Plot(titlestr, X, outname, label, outdir, grid = True,
         savePDF = True, savePNG = False, datestamp = True):
    
    mpl.rcParams['xtick.top'] = False
    mpl.rcParams['xtick.bottom'] = True
    mpl.rcParams['ytick.right'] = False
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'
    
    mpl.rc('font',**{'size': 10})
    mpl.rc('legend',**{'fontsize': 5.5})
    mpl.rc("axes", linewidth = 0.5)    
    
    plt.rc('font', **{'family' : 'sans-serif', 'sans-serif' : ['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42  
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['mathtext.fontset'] = 'cm'
    fontparams = {'text.latex.preamble': [r'\usepackage{cmbright}', r'\usepackage{amsmath}']}
    mpl.rcParams.update(fontparams)
    
    ######################################################################################
    # set up figure
    fWidth, fHeight, lFrac, rFrac, bFrac, tFrac =\
        getFigureProps(width = 5.5, height = 4.0,
                       lFrac = 0.16, rFrac = 0.95, bFrac = 0.17, tFrac = 0.95)
    f, ax1 = plt.subplots(1)
    f.set_size_inches(fWidth, fHeight)    
    f.subplots_adjust(left = lFrac, right = rFrac)
    f.subplots_adjust(bottom = bFrac, top = tFrac)
    ######################################################################################
    
    major_x_ticks = np.arange(-10.0, 12.5, 2.0)
    minor_x_ticks = np.arange(-10.0, 12.5, 1.0)
    ax1.set_xticks(major_x_ticks)
    ax1.set_xticks(minor_x_ticks, minor = True)
    
    major_y_ticks = np.arange(0.0, 0.22, 0.1)
    minor_y_ticks = np.arange(0.0, 0.22, 0.05)
    ax1.set_yticks(major_y_ticks)
    ax1.set_yticks(minor_y_ticks, minor = True)
    
    labelfontsize = 8.0
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)
    
    xticks = plt.getp(plt.gca(), 'xticklines')
    yticks = plt.getp(plt.gca(), 'yticklines')
    ax1.tick_params('both', length = 2.5, width = 0.5, which = 'major', pad = 2.0)
    ax1.tick_params('both', length = 1.75, width = 0.35, which = 'minor', pad = 2.0)
    ######################################################################################
    # labeling
    plt.title(titlestr)
    ax1.set_xlabel(r'position $x$', fontsize = 8.0)
    ax1.set_ylabel(r'$p(x,t\, |\, x_0,t_0)$', fontsize = 8.0)
    ax1.xaxis.labelpad = 2.0
    ax1.yaxis.labelpad = 4.0
    
    ######################################################################################
    # plotting

    xVals = np.linspace(-10.0, 10.0, 1000)
    
    ax1.plot(xVals, FPE_free(xVals, x0 = 2.0, t = 3.0),
             color = '#666666',
             alpha = 1.0,
             lw = 1.0,
             label = r'FPE$(x, t = 3 \, | \, x_0 = 2, t_0 = 0)$')

    counts, bins = np.histogram(X, bins = 40, normed = True)
    binCenters = [0.5 * (bins[k] + bins[k+1]) for k in range(len(counts))]
    binWidth = bins[1] - bins[0]
    ax1.scatter(binCenters, counts,
                    s = 10.0,
                    marker = 'o',
                    facecolors = 'None',
                    edgecolors = 'k',
                    linewidth = 0.65,
                    label = label,
                    zorder = 3)
                 
    ax1.plot([x0Value, x0Value], [-0.2, 1.0],
             lw = 0.5,
             color = '#CCCCCC',
             alpha = 1.0,
             dashes = [6.0, 3.0],
             zorder = 1)
             
    ######################################################################################
    # legend
    leg = ax1.legend(loc = 'upper left',
                     handlelength = 2.0,
                     fontsize = 6.0,
                     scatterpoints = 1,
                     markerscale = 1.0,
                     ncol = 1)
    leg.draw_frame(False)
    ######################################################################################
    # set plot range
    ax1.set_xlim(-6.25, 10.25)
    ax1.set_ylim(-0.01, 0.205)
    ######################################################################################
    # grid options
    if (grid):
        ax1.grid(color = 'gray', alpha = 0.15, lw = 0.3, linestyle = 'dashed', dashes = [4.0, 1.0])
    
    ######################################################################################
    # inlet (ax2 object)
    ax2 = f.add_axes([0.48, 0.25, 0.21, 0.22])
    
    ax2.tick_params(direction = 'in', which = 'both')
        
    for axis in ['top', 'bottom', 'left', 'right']:
        ax2.spines[axis].set_linewidth(0.35)
        
    ax2.tick_params('both', length = 1.5, width = 0.35, which = 'major', pad = 3.0)
    ax2.tick_params('both', length = 1.0, width = 0.25, which = 'minor', pad = 3.0)
    
    ax2.tick_params(axis='x', which='major', pad = 1.0)
    ax2.tick_params(axis='y', which='major', pad = 1.0, zorder = 10)
    
    labelfontsize = 4.0
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)
        
    major_x_ticks = np.arange(-10.0, 15.1, 5.0)
    minor_x_ticks = np.arange(-10.0, 15.1, 1.0)
    ax2.set_xticks(major_x_ticks)
    ax2.set_xticks(minor_x_ticks, minor = True)
  
    major_y_ticks = np.arange(-0.1, 0.25, 0.05)
    minor_y_ticks = np.arange(-0.1, 2.25, 0.01)
    ax2.set_yticks(major_y_ticks)
    ax2.set_yticks(minor_y_ticks, minor = True)
    
    
    residuals = [(counts[i] - FPE_free(binCenters[i], 2, 3.0)) \
                 for i in range(len(binCenters))]
    
    ax2.plot(binCenters, residuals,
             lw = 0.5,
             color = 'k')    
     
    ax2.set_xlabel(r'position $x$', fontsize = 4.0)
    ax2.set_ylabel(r'res. $\Delta p(x,t)$', fontsize = 4.0)
    ax2.xaxis.labelpad = 0.0
    ax2.yaxis.labelpad = -1.0
                    
    ax2.set_xlim(-6.0, 10.0)
    ax2.set_ylim(-0.05, 0.05)

    ax2.grid(color = 'gray', alpha = 0.25, lw = 0.35, linestyle = 'dashed', dashes = [4.0, 3.0])
    
    ######################################################################################
    # save to file
    if (datestamp):
        outname += '_' + now
    if (savePDF):
        f.savefig(os.path.join(outdir, outname) + '.pdf', dpi = 300, transparent = False)
    if (savePNG):
        f.savefig(os.path.join(outdir, outname) + '.png', dpi = 600, transparent = False)
    plt.clf()
    plt.close()
    return None

def FPE_free(x, x0, t):
    return np.exp(-(x-x0) ** 2 / (4 * t)) / np.sqrt(4.0 * np.pi * t)

if __name__ == '__main__':

    x0Value = 2.0
    
    A = np.genfromtxt(os.path.join(RAWDIR, '04_A_freeDiffusion_dt_1e-3_m_10000.txt'))
    B = np.genfromtxt(os.path.join(RAWDIR, '04_B_freeDiffusion_dt_1e-2_m_10000.txt'))
    C = np.genfromtxt(os.path.join(RAWDIR, '04_C_freeDiffusion_dt_1e-1_m_10000.txt'))
    D = np.genfromtxt(os.path.join(RAWDIR, '04_D_freeDiffusion_dt_1e0_m_10000.txt'))
        
    Plot(titlestr = '', 
         X = A,
         outname = '04_A_freeDiffusion_dt_1e-3', 
         label = r'$dt = 10^{-3}$',
         outdir = OUTDIR, 
         grid = False)
            
    Plot(titlestr = '', 
         X = B,
         outname = '04_B_freeDiffusion_dt_1e-2', 
         label = r'$dt = 10^{-2}$',
         outdir = OUTDIR, 
         grid = False)

    Plot(titlestr = '', 
         X = C,
         outname = '04_C_freeDiffusion_dt_1e-1', 
         label = r'$dt = 10^{-1}$',
         outdir = OUTDIR, 
         grid = False)
            
    Plot(titlestr = '', 
         X = D,
         outname = '04_D_freeDiffusion_dt_1e0', 
         label = r'$dt = 1$',
         outdir = OUTDIR, 
         grid = False)
    

    
   




