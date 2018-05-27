#!/usr/bin/python
# -*- coding: utf-8 -*-
##########################################################################################
# author: Nikolas Schnellbaecher
# contact: khx0@posteo.net
# date: 2018-05-27
# file: 02_analyze_freeDiffusionReflectiveBC.py
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
from scipy import stats
import scipy

from Langevin import getMoments

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
    
def Plot(titlestr, X, MSD, outname, outdir, grid = True,
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
                       lFrac = 0.16, rFrac = 0.95, bFrac = 0.17, tFrac = 0.92)
    f, ax1 = plt.subplots(1)
    f.set_size_inches(fWidth, fHeight)    
    f.subplots_adjust(left = lFrac, right = rFrac)
    f.subplots_adjust(bottom = bFrac, top = tFrac)
    ######################################################################################
    
    major_x_ticks = np.arange(0.0, 15.0, 5.0)
    minor_x_ticks = np.arange(0.0, 15.0, 1.0)
    ax1.set_xticks(major_x_ticks)
    ax1.set_xticks(minor_x_ticks, minor = True)
    
    major_y_ticks = np.arange(0.0, 0.55, 0.2)
    minor_y_ticks = np.arange(0.0, 0.55, 0.1)
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
    ax1.yaxis.labelpad = 5.0
    ######################################################################################
    # cmap settings
    sampleTimes = [0.5, 1.0, 3.0, 5.0, 10.0]
    ColorMap = cmx.viridis
    cNorm = mpl.colors.LogNorm(vmin = sampleTimes[0], vmax = 12.2)
    scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = ColorMap)
    print "Colormap colornorm limits =", scalarMap.get_clim()
    ######################################################################################
    # plotting
    nSamples = len(sampleTimes)
    
    xVals = np.linspace(0.0, 20.0, 1000)
    
    labels = [r'$t = 0.5$',
              r'$t = 1$',
              r'$t = 3$',
              r'$t = 5$',
              r'$t = 10$']
    
    for i in range(nSamples):
        
        RGBcolorValue = scalarMap.to_rgba(sampleTimes[i])
        ax1.plot(xVals, FPE_reflective(xVals, x0 = x0Value, t = sampleTimes[i]),
                 color = RGBcolorValue,
                 lw = 1.0)
        
        counts, bins = np.histogram(X[i, :], bins = 30, normed = True)
        binCenters = [0.5 * (bins[k] + bins[k+1]) for k in range(len(counts))]
        binWidth = bins[1] - bins[0]
        ax1.scatter(binCenters, counts,
                    s = 10.0,
                    marker = 'o',
                    facecolors = 'None',
                    edgecolors = RGBcolorValue,
                    linewidth = 0.65,
                    label = labels[i],
                    zorder = 3)
                 
    ax1.plot([x0Value, x0Value], [-0.2, 1.0],
             lw = 0.5,
             color = '#CCCCCC',
             alpha = 1.0,
             dashes = [6.0, 3.0],
             zorder = 1)

    ######################################################################################
    ######################################################################################
    # dummy plot
    handles = []
    
    p, = ax1.plot([-999.0],[-999.0],
                  lw = 1.25,
                  color = 'k')

    handles.append(p) 
    labels = [r'FPE (analytical)']
    Dleg = plt.legend(handles, 
                      labels, 
                      loc = 'upper left',
                      bbox_to_anchor = [0.0, 1.11], 
                      handlelength = 2.0)
    Dleg.draw_frame(False)
    plt.gca().add_artist(Dleg)
    
    for k, spine in ax1.spines.items():  #ax.spines is a dictionary
        spine.set_zorder(10)
    
    ######################################################################################
    ######################################################################################
    
    ######################################################################################
    # legend
    leg = ax1.legend(loc = 'upper right',
                     handlelength = 1.0, 
                     scatterpoints = 1,
                     markerscale = 1.5,
                     ncol = 1)
    for i, legobj in enumerate(leg.legendHandles):
        legobj.set_linewidth(1.0)
    leg.draw_frame(False)
    ######################################################################################
    # set plot range
    ax1.set_xlim(0.0, 12.3)
    ax1.set_ylim(-0.015, 0.45)
    ######################################################################################
    # grid options
    if (grid):
        ax1.grid(color = 'gray', alpha = 0.15, lw = 0.3, linestyle = 'dashed', dashes = [2.0, 1.0])
    
    ######################################################################################
    # inlet (ax2 object) first moments (\mu)
    
    ax2 = f.add_axes([0.67, 0.32, 0.26, 0.23])
    
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

    major_x_ticks = np.arange(0.0, 12.1, 5.0)
    minor_x_ticks = np.arange(0.0, 12.1, 1.0)
    ax2.set_xticks(major_x_ticks)
    ax2.set_xticks(minor_x_ticks, minor = True)
  
    major_y_ticks = np.arange(0.0, 5.1, 1.0)
    minor_y_ticks = np.arange(0.0, 5.1, 0.5)
    ax2.set_yticks(major_y_ticks)
    ax2.set_yticks(minor_y_ticks, minor = True)
    
    firstMoments = np.mean(X, axis = 1)
    
    ax2.scatter(sampleTimes, firstMoments,
                s = 6.0,
                marker = 'o',
                facecolors = 'None',
                edgecolors = 'k',
                linewidth = 0.35,
                zorder = 3)
                    
    xVals = np.linspace(0.0001, 12.0, 1500)
    yVals = [ Mean_ReflectiveBC(t, 2.0) for t in xVals]
    
    ax2.plot(xVals, yVals,
             lw = 0.75,
             alpha = 1.0,
             color = '#666666',
             zorder = 1,
             label = r'$\langle x \rangle(t)$')

    leg = ax2.legend(bbox_to_anchor = [-0.02, 1.02],
                     loc = 'upper left',
                     handlelength = 2.0, 
                     scatterpoints = 1,
                     fontsize = 4.0,
                     markerscale = 1.5,
                     ncol = 1)
    leg.draw_frame(False)
            
    ax2.set_xlabel(r'time $t$', fontsize = 4.0)
    ax2.set_ylabel(r'$\langle x \rangle (t)$', fontsize = 4.0)
    ax2.xaxis.labelpad = 0.0
    ax2.yaxis.labelpad = 2.0
                    
    ax2.set_xlim(-0.5, 11.5)
    ax2.set_ylim(1.8, 4.35)

    ax2.grid(color = 'gray', alpha = 0.25, lw = 0.35, linestyle = 'dashed', dashes = [4.0, 3.0])
    
    ######################################################################################
    # inlet (ax3 object) MSD plot
    
    ax3 = f.add_axes([0.43, 0.62, 0.31, 0.27])
    
    for axis in ['top', 'bottom', 'left', 'right']:
        ax3.spines[axis].set_linewidth(0.35)
        
    ax3.tick_params('both', length = 1.5, width = 0.35, which = 'major', pad = 3.0)
    ax3.tick_params('both', length = 1.0, width = 0.25, which = 'minor', pad = 3.0)
    
    ax3.tick_params(axis='x', which='major', pad = 1.0)
    ax3.tick_params(axis='y', which='major', pad = 1.0, zorder = 10)
    
    labelfontsize = 4.0
    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)
        
    major_x_ticks = np.arange(0.0, 12.1, 5.0)
    minor_x_ticks = np.arange(0.0, 12.1, 1.0)
    ax3.set_xticks(major_x_ticks)
    ax3.set_xticks(minor_x_ticks, minor = True)

    major_y_ticks = np.arange(0.0, 25.1, 5.0)
    minor_y_ticks = np.arange(0.0, 25.1, 1.0)
    ax3.set_yticks(major_y_ticks)
    ax3.set_yticks(minor_y_ticks, minor = True)
    
    ax3.scatter(sampleTimes, MSD,
                s = 6.0,
                marker = 'o',
                facecolors = 'None',
                edgecolors = 'k',
                linewidth = 0.35,
                zorder = 3)      
                    
    xVals = np.linspace(0.0001, 12.0, 1500)
    yVals = [ MSD_ReflectiveBC(t, 2.0) for t in xVals]    
    yValsFree = [2.0 * x for x in xVals]              
                 
    ax3.plot(xVals, yVals,
             lw = 0.75,
             alpha = 1.0,
             color = '#666666',
             zorder = 1,
             label = r'MSD$(t)$')

    ax3.plot(xVals, yValsFree,
             lw = 0.75,
             alpha = 1.0,
             color = '#666666',
             label = r'$\sim 2Dt$',
             dashes = [2.0, 1.0])
    
    leg = ax3.legend(bbox_to_anchor = [0.45, 0.42],
                     loc = 'upper left',
                     fontsize = 4.0,
                     handlelength = 2.0, 
                     scatterpoints = 1,
                     markerscale = 1.0,
                     ncol = 1)
    leg.draw_frame(False)
                
    ax3.set_xlabel(r'time $t$', fontsize = 4.0)
    ax3.set_ylabel(r'MSD$(t)$', fontsize = 4.0)
    ax3.xaxis.labelpad = 1.0
    ax3.yaxis.labelpad = 1.0
                  
    ax3.set_xlim(0.0, 11.5)
    ax3.set_ylim(0.0, 10.5)
             
    ax3.grid(color = 'gray', alpha = 0.25, lw = 0.35, linestyle = 'dashed', dashes = [4.0, 3.0])
    
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
    
def FPE_reflective(x, x0, t):
    return FPE_free(x, x0, t) + FPE_free(x, -x0, t)
    
def Mean_ReflectiveBC(t, x0):
    tmp = np.exp(-x0 ** 2 / (4.0 * t)) * np.sqrt(4.0 * t / np.pi)
    return tmp + x0 * scipy.special.erf(x0 / np.sqrt(4.0 * t))

def MSD_ReflectiveBC(t, x0):
    c = 4.0 * t
    tmp = 2.0 * t + x0 ** 2
    tmp -= (2.0 * np.exp(- x0 ** 2 / c) * np.sqrt(t/np.pi) + x0 * \
           scipy.special.erf(x0 / (2.0 * np.sqrt(t))) ) ** 2
    return tmp

if __name__ == '__main__':

    filename = '02_freeDiffusion_reflectiveBC_m_100000_dt_1e-03.txt'
    outname = filename.split('.')[0]
    
    X = np.genfromtxt(os.path.join(RAWDIR, filename))
    print("X.shape =", X.shape)
    
    x0Value = 2.0
    sampleTimes = [0.5, 1.0, 3.0, 5.0, 10.0]
    
    firstMoments, MSD = getMoments(X, sampleTimes)

    print firstMoments
    
    print MSD
    
    Plot(titlestr = '', 
         X = X,
         MSD = MSD, 
         outname = outname,
         outdir = OUTDIR, 
         grid = False)

   




