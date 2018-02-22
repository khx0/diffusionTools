#!/usr/bin/python
# -*- coding: utf-8 -*-
##########################################################################################
# author: Nikolas Schnellbaecher
# contact: khx0@posteo.net
# date: 2018-02-22
# file: test_StokesEinstein.py
##########################################################################################

import os
import sys
import numpy as np
import unittest

from StokesEinstein import D_StokesEinstein

class StokesEinsteinTest(unittest.TestCase):
    
    """
    Test cases for the StokesEinstein function.
    """
    
    def test_01(self):
    
        """
        A common rough estimate for the diffusion coefficient according to 
        the Stokes Einstein relation gives
        D = 100.0 (um)^2 / s for a r = 1 nm sized particle
        in water at room temperature. 
        This is in fact only a very rought estimate, since the true value
        is D = 214.38 (um)^2 / s as shown in this first test below.
        """
        
        # set the temperature to 20 degree Celsius
        # absolute zero in Kelvin corresponds to -273.15 degree Celsius
        temp = 293.15       # in Kelvin (20 degree Celsius)
        eta = 1.0016        # in m Pa * s (for water at temp = 293.15 K)
        radius = 1.0        # in nm
        
        Dref = 214.3767049
        D = D_StokesEinstein(temp, eta, radius)
        
        self.assertTrue(np.isclose(D, Dref))
        
    def test_02(self):
    
        """
        Stokes Einstein estimate for
        spherical particles of radius = (60 - 75) nm 
        at two different temperatures in water.
        This estimate corresponds roughly to HIV1 virions.
        See also Milo R., Phillips R. - Cell Biology by the Numbers [2015].
        """
        
        ##################################################################################
        # set the temperature to 20 degree Celsius
        # absolute zero in Kelvin corresponds to -273.15 degree Celsius
        temp = 293.15           # in Kelvin (20 degree Celsius)
        eta = 1.0016            # in m Pa * s (for water at temp = 293.15 K)
        
        radius1 = 60.0          # in nm
        radius2 = 75.0          # in nm
        
        Dref1 = 3.57294508166   # in (um)^2 / s
        Dref2 = 2.85835606533   # in (um)^2 / s
        
        D1 = D_StokesEinstein(temp, eta, radius1)
        D2 = D_StokesEinstein(temp, eta, radius2)
        
        self.assertTrue(np.isclose(D1, Dref1))
        self.assertTrue(np.isclose(D2, Dref2))
        
        ##################################################################################
        # next we go to 37 degree Celsius
        temp = 310.15       # in Kelvin
        eta = 0.6913        # in m Pa * s (for water at temp = 310.15 K)
        
        Dref1 = 5.4769148232    # in (um)^2 / s
        Dref2 = 4.38153185856   # in (um)^2 / s
        
        D1 = D_StokesEinstein(temp, eta, radius1)
        D2 = D_StokesEinstein(temp, eta, radius2)
        
        self.assertTrue(np.isclose(D1, Dref1))
        self.assertTrue(np.isclose(D2, Dref2))

if __name__ == '__main__':

    unittest.main()




    
    
    
    
    
    
    
    
    
    
    
    
