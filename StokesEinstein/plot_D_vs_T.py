#!/usr/bin/python
# -*- coding: utf-8 -*-
##########################################################################################
# author: Nikolas Schnellbaecher
# contact: khx0@posteo.net
# date: 2018-02-22
# file: plot_D_vs_T.py
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
import matplotlib.cm as cm
from matplotlib import gridspec
from matplotlib import ticker

from StokesEinstein import D_StokesEinstein
from StokesEinstein import viscTable

mpl.ticker._mathdefault = lambda x: '\\mathdefault{%s}'%x

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

now = datetime.datetime.now()
now = "%s-%s-%s" %(now.year, str(now.month).zfill(2), str(now.day).zfill(2))

BASEDIR = os.path.dirname(os.path.abspath(__file__))
RAWDIR = os.path.join(BASEDIR, 'raw')
OUTDIR = os.path.join(BASEDIR, 'out')

ensure_dir(OUTDIR)

def getFigureProps(width, height):
    '''
    Specify widht and height in cm
    '''
    lFrac, rFrac = 0.20, 0.90
    bFrac, tFrac = 0.20, 0.90
    axesWidth = width / 2.54 # convert to inches
    axesHeight = height / 2.54 # convert to inches
    fWidth = axesWidth / (rFrac - lFrac)
    fHeight = axesHeight / (tFrac - bFrac)
    return fWidth, fHeight, lFrac, rFrac, bFrac, tFrac

def Plot(titlestr, X, params, outname, outdir, pColors,
         grid = True, savePDF = True, savePNG = False, datestamp = True):
             
    mpl.rcParams['ytick.left'] = True
    mpl.rcParams['xtick.top'] = False
    mpl.rcParams['xtick.bottom'] = True
    mpl.rcParams['ytick.right'] = False
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'
    
    mpl.rc('font',**{'size': 10})
    mpl.rc('legend',**{'fontsize': 5.0})
    mpl.rc("axes", linewidth = 0.5)    
    
    # plt.rc('font', **{'family' : 'sans-serif', 'sans-serif' : ['Myriad Pro']})
    plt.rc('font', **{'family' : 'sans-serif', 'sans-serif' : ['Arial']})
    plt.rcParams['pdf.fonttype'] = 42  
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['mathtext.fontset'] = 'cm'
    fontparams = {'text.latex.preamble': [r'\usepackage{cmbright}', r'\usepackage{amsmath}']}
    mpl.rcParams.update(fontparams)      
    
    ######################################################################################
    # set up figure
    fWidth, fHeight, lFrac, rFrac, bFrac, tFrac =\
        getFigureProps(width = 3.5, height = 2.5)
    f, ax1 = plt.subplots(1)
    f.set_size_inches(fWidth, fHeight)    
    f.subplots_adjust(left = lFrac, right = rFrac)
    f.subplots_adjust(bottom = bFrac, top = tFrac)
    ######################################################################################
    
    major_x_ticks = np.arange(15.0, 40.0, 5.0)
    minor_x_ticks = np.arange(15.0, 40.0, 1.0)
    ax1.set_xticks(major_x_ticks)
    ax1.set_xticks(minor_x_ticks, minor = True)
    
    major_y_ticks = np.arange(0.0, 11.0, 1.0)
    minor_y_ticks = np.arange(0.0, 11.0, 0.5)
    ax1.set_yticks(major_y_ticks)
    ax1.set_yticks(minor_y_ticks, minor = True)
        
    labelfontsize = 5.0
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)
    
    ax1.tick_params('both', length = 2.5, width = 0.5, which = 'major', pad = 3.0)
    ax1.tick_params('both', length = 1.5, width = 0.25, which = 'minor', pad = 3.0)
    
    ax1.tick_params(axis = 'x', which = 'major', pad = 1.0)
    ax1.tick_params(axis = 'y', which = 'major', pad = 1.0, zorder = 10)
    ######################################################################################
    # labeling
    plt.title(titlestr)
    ax1.set_xlabel(r'temperature $\, T \,\,\, [^{\circ}\mathrm{C}]$', fontsize = 6.0)
    ax1.set_ylabel('diffusion coefficient\n$D \,\,\, [\mu\mathrm{m}^2/\mathrm{s}]$',
                   fontsize = 6.0,
                   multialignment = 'left')
    
    ax1.xaxis.labelpad = 3.0
    ax1.yaxis.labelpad = 3.0
    ######################################################################################
    # plot data
        
    ax1.fill_between(X[:, 0], X[:, 1], X[:, 2], 
                     color = pColors[2],
                     alpha = 0.2,
                     linewidth = 0.0,
                     zorder = 1,
                     label = r'Stokes-Einstein')
    
    ax1.plot(X[:, 0], X[:, 1],
             color = pColors[0],
             lw = 0.5,
             alpha = 1.0,
             clip_on = True,
             zorder = 1)

    ax1.plot(X[:, 0], X[:, 2],
             color = pColors[0],
             lw = 0.5,
             alpha = 1.0,
             clip_on = True,
             zorder = 1)

    ax1.plot([10.0, 50.0], [params[0], params[0]],
             lw = 0.5,
             alpha = 1.0,
             color = 'k',
             label = r'$D = 3.35 \, \mu\mathrm{m}^2/\mathrm{s} \,$ (experiment)')
             
    ax1.scatter([20.0, 20.0], [X[4, 1], X[4, 2]],
                s = 3.0,
                marker = 'o',
                alpha = 1.0,
                lw = 0.5,
                facecolor = 'C0',
                edgecolor = 'C0', 
                zorder = '5',
                clip_on = False)

    ax1.scatter([37.0, 37.0], [X[-4, 1], X[-4, 2]],
                s = 3.0,
                marker = 'o',
                alpha = 1.0,
                lw = 0.5,
                facecolor = 'C0',
                edgecolor = 'C0', 
                zorder = '5',
                clip_on = False)
                
    ax1.plot([20.0, 20.0], [-2, X[4, 1]],
             lw = 0.5,
             alpha = 1.0,
             color = 'C0',
             dashes = [1.0, 0.5],
             zorder = 1)
             
    ax1.plot([37.0, 37.0], [-2, X[-4, 1]],
             lw = 0.5,
             alpha = 1.0,
             color = 'C0',
             dashes = [1.0, 0.5],
             zorder = 1)
        
    ######################################################################################
    # legend
    leg = ax1.legend(loc = 'upper left',
                     handlelength = 1.75, 
                     scatterpoints = 1,
                     markerscale = 1.0,
                     ncol = 1)
    leg.draw_frame(False)
    ######################################################################################
    #ax1.set_axisbelow(False)
    for k, spine in ax1.spines.items():  #ax.spines is a dictionary
        spine.set_zorder(10)               
    ######################################################################################
    # set plot range and scale
    ax1.set_xlim(16.5, 39.9)
    ax1.set_ylim(2.4, 6.2)
    ax1.set_axisbelow(False)
    ######################################################################################
    # grid options
    if (grid):
        ax1.grid(color = 'gray', linestyle = '-', alpha = 0.2, which = 'major', linewidth = 0.4)
        ax1.grid('on')
        ax1.grid(color = 'gray', linestyle = '-', alpha = 0.05, which = 'minor', linewidth = 0.2)
        ax1.grid('on', which = 'minor')
    ######################################################################################
    # save to file
    if (datestamp):
        outname += '_' + now
    if (savePDF): # save to file using pdf backend
        f.savefig(os.path.join(outdir, outname) + '.pdf', dpi = 300, transparent = True);
    if (savePNG):
        f.savefig(os.path.join(outdir, outname) + '.png', dpi = 600, transparent = False);
    ######################################################################################
    # close handles
    plt.clf()
    plt.close()
    return outname
       
DReferenceValue = 3.35 # in (um)^2 / s

ABSOLUTEZERO = 273.15  
    
radius1 = 60.0       # in nm
radius2 = 75.0       # in nm 
                 
if __name__ == '__main__':

    tempValues = viscTable[:, 0] + ABSOLUTEZERO
    
    X = np.zeros((len(tempValues), 3))
    
    for i, temp in enumerate(tempValues):
        
        eta = viscTable[i, 1]
        X[i, 0] = viscTable[i, 0]
        X[i, 1] = D_StokesEinstein(temp, eta, radius1)
        X[i, 2] = D_StokesEinstein(temp, eta, radius2)
             
    colorVals = ['C0', '#CCCCCC', 'C0', '#666666', 'C0']
    
    outname = Plot(titlestr = '',
                   X = X,
                   params = [DReferenceValue],
                   outname = 'D_vs_T_figure',
                   outdir = OUTDIR, 
                   pColors = colorVals,
                   grid = False)
    
    cmd = 'pdf2svg ' + os.path.join(OUTDIR, outname + '.pdf') + \
          ' ' + os.path.join(OUTDIR, outname + '.svg')
    print cmd
    os.system(cmd)

    
    

    



