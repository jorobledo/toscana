#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------1---------2---------3---------4---------5---------6---------7---------
"""Module for basic data treatment of disordered materials data.

Module: toscana.py

This module contains functions and variables that are useful for the basic
data treatment total scattering experiments.

The functions are thougth to use D4 data, but in fact they could be used for
a more general case.

For a listing of the functions in the module:
    dir(toscana)

For a detailed help of all functions:
    help(toscana)

Date: Wed Dec 30 23:47:15 2020
Author: Gabriel Cuello
---------------------------------------
"""
print('ToScaNA (TOtal SCAttering Neutron Analysis)')
print('(by Gabriel Cuello, ILL, Dec 2023)')
print('Module imported')

#--------1---------2---------3---------4---------5---------6---------7---------8
import sys,os
#from run import run_cmd
import glob
import subprocess
from datetime import datetime
import calendar       # To find the weekday
import h5py
import math
import cmath
import numpy as np                   # NumPy library
from scipy import integrate          # For integration
from scipy.signal import find_peaks
import lmfit as lm                   # For fitting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import statistics as st
#--------1---------2---------3---------4---------5---------6---------7---------8


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#
#  Miscelaneous Calculations
#
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


#--------1---------2---------3---------4---------5---------6---------7---------8
def getCylVolume(diameter=5.0,height=50.0):
    """
    Calculate the volume of a cylinder.

    The units of these two distances must be the same, and the result is given
    in the same unit3 (if distances are in mm, the volume will be in mm3).

    Parameters
    ----------
    - diameter : float, optional
        The diameter of the cylinder. The default is 5.0.
    - height : float, optional
        The height of the cylinder. The default is 50.0.

    Returns
    -------
    float : The volume of the cylinder.

    Author: Gabriel Cuello
    Date: Oct 29 2021
--------------------------------------------------------------------------------
    """
    return np.pi * (diameter/2.0)**2 * height
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def stop(message):
    """
    Stop the program at this point.
    This is not really useful for the Jupyter versions of the scripts, but it
    is for the python versions.

    Parameters
    ----------
    -  answer: string (from the console)
        'n' or anything else
    - message: string containing a message to be printed just before asking
        whether continuing or not.

    Returns
    -------
    Nothing: no output

    Author: Gabriel Cuello
    Date: Oct 25 2021
--------------------------------------------------------------------------------
    """
    print ()
    print (30*'<>')
    print (message)
    answer = input("Do you want to continue? (y/n) :")
    if answer == 'n':
        sys.exit("You stopped the program. Bye!")
    return None
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def ang2q(x,wlength=0.5):
    """
    Convert a numpy array of angles in the corresponding Q values, using the
    Bragg law
         Q = 4 pi / lambda sin(angle/2 / 180 * pi)
    where angle is the scattering angle (2theta) in degrees.

    Parameters
    ----------
    - x: A list with the angles (in degrees).
    - wlenght: The wavelength (in A).

    Returns
    -------
    - array, floats: A numpy array with the Q values (in 1/A).

    Author: Gabriel Cuello
    Date: Oct 24, 2021
--------------------------------------------------------------------------------
    """
    if wlength <= 0:
        stop("Wrong wavelength in ang2q function")
    q = 4.0*np.pi/wlength * np.sin(np.array(x)/2.0*np.pi/180.0)
    return q
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def q2ang (x,wlength=0.5):
    """
    Converts a numpy array of Q values in the corresponding angles, using the
    Bragg law
             angle = 2 arcsin(Q lambda / 4 pi) / pi * 180
    where angle is the scattering angle (2theta) in degrees.

    Parameters
    ----------
    - x: float numpy array
        Q values (in 1/Å)
    - wlenght: float positive
        The wavelength (in Å).

    Returns
    -------
    - array, float: numpy array with the angles (in degrees).

    Author: Gabriel Cuello
    Date: Oct 24, 2021
--------------------------------------------------------------------------------
    """
    if wlength <= 0:
        stop("Wrong wavelength in q2ang function")
    ang = 360.0/np.pi * np.arcsin(np.array(x)*wlength/4.0/np.pi)
    return ang
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------
def ratio(y1,e1,y2,e2):
    """
    Calculate the ratio between two sets of data. Both sets must have the same
    length, i.e., the same number of points.
         ratio = y1/y2
         error = sqrt((y2 * e1)**2 + (y1 * e2)**2) / y2**2

    Parameters
    ----------
    - y1,e1 : lists with numerator and its error
    - y2,e2 : lists with denominator and its error

    Returns
    -------
    - lists, float: two lists with the ratio and its error

    Date: Thu Jan 01 2021
    Author: Gabriel Cuello
    Modified: 13/05/22 Use of the zip function to simultaneously iterate lists
---------------------------------------
    """
    y_rat = []
    e_rat = []

    for y1_val, e1_val, y2_val, e2_val in zip(y1, e1, y2, e2):
        if y2_val != 0:
            ratio = y1_val / y2_val
            error = np.sqrt((y2_val * e1_val)**2 + (y1_val * e2_val)**2) / y2_val**2
        else:
            ratio = 0.0
            error = 0.0

        y_rat.append(ratio)
        e_rat.append(error)
    return y_rat,e_rat
#--------1---------2---------3---------4---------5---------6---------7---------


#--------1---------2---------3---------4---------5---------6---------7---------
def smooth_curve(x, y, smoothing_factor):
    '''
    This function smooths a function using a moving average-window.

    Parameters
    ----------
    x : array
        Array x values.
    y : array
        Array of y values.
    smoothing_factor : float
        Smoothing factor between 0 and 1. A smaller smoothing factor will result
        in less smoothing, while a larger factor will result in more aggressive
        smoothing.

    Returns
    -------
    smoothed_y : array
        Array smoothed y values.

    Date: Thu Jan 01 2021
    Author: Gabriel Cuello
    '''
    smoothed_y = []
    window_size = int(smoothing_factor * len(y))

    for i in range(len(y)):
        start = max(0, i - window_size // 2)
        end = min(len(y), i + window_size // 2 + 1)
        smoothed_value = np.mean(y[start:end])
        smoothed_y.append(smoothed_value)

    return np.array(smoothed_y)
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def get_chi(y1,y2):
    '''
    This function calculates the chi between 2 sets of y-values.

    Parameters
    ----------
    y1 : array
        Array of values of the first set of y-values.
    y2 : array
        Array of values of the second set of y-values.

    Both arrays are supposed to have the same x-values and consequently the
    same number of points

    Returns
    -------
    chi : float
        Value of chi between the 2 sets of y-values.
    '''
    return np.sqrt(np.mean((np.array(y1) - np.array(y2))**2))
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def fit_and_find_extremum(x, y):
    '''
The fit_and_find_extremum function fits a second-degree polynomial
to the given points, calculates the derivative, and finds the x value
where the derivative is 0. It then returns the extremum x and y values.
    '''
    # Fit a second-degree polynomial
    coeffs = np.polyfit(x, y, 2)
    polynomial = np.poly1d(coeffs)

    # Calculate the derivative of the polynomial
    derivative = polynomial.deriv()

    # Find the x value where the derivative is 0
    extremum_x = derivative.r
    extremum_y = polynomial(extremum_x)
    fit = np.polyval(coeffs,x)
    return extremum_x, extremum_y, fit
#--------1---------2---------3---------4---------5---------6---------7---------


#--------1---------2---------3---------4---------5---------6---------7---------
def wsum2(w1,data1,w2,data2):
    '''
    Sum (or subtract) two sets of values (y and error).
    Depending on the parameters w1 and w2, different operations are done.
    w1 and w2 are the weights for the sets 1 (y1,e1) and 2 (y2,e2).

    Case (1): both weights are zero (w1=w2=0)
        The function returns the sum weighted with the inverse of the square errors.
    Case (2): w1=0 and w2!=0
        The function returns the set 2 multiplied by w2
    Case (3): w1!=0 and w2=0
        If w1 is not in the range (0,1], error.
        Otherwise, the function returns the weighted sum with w2=1-w1
    Case (4): Both weights are different from zero (w1!=0 and w2!=0)
        The functions returns simply w1*y1+w2*y2.
        Note that the weights can also be negative, so this case includes the subtraction.

    Parameters
    ----------
    - w1, data1: weight and data for set 1
    - w2, data2: weight and data for set 2
        Data are given as a 3-column matrix x,y,e

    Returns
    -------
    matrix: the resulting matrix in the same format as the input matrix

    Date: Thu Jan 01 2021
    Author: Gabriel Cuello
---------------------------------------
    '''
    x1 = data1[:,0]
    y1 = data1[:,1]
    e1 = data1[:,2]
#    x2 = data1[:,0]
    y2 = data2[:,1]
    e2 = data2[:,2]
    if (len(y1)!=len(y2)):
        print('--- Error in the binary sum (wsum2).')
        print('--- The input vectors have not the same length.')
        return
    length = len(y1)
    ysum = []
    esum = []
    if ((w1==0) and (w2==0)):
        for i in range(length):
            w = 0
            sqerr = e1[i]**2+e2[i]**2
            if (sqerr!=0): w = e2[i]**2/sqerr
            ysum.append(w*y1[i]+(1-w)*y2[i])
            esum.append(np.sqrt(w**2*e1[i]**2+(1.0-w)**2*e2[i]**2))
    elif (w1==0):
        for i in range(length):
            ysum.append(w2*y2[i])
            esum.append(w2*e2[i])
    elif (w2==0):
        if ((w1>0) and (w1<=1)):
            for i in range(length):
                ysum.append(w1*y1[i]+(1-w1)*y2[i])
                esum.append(np.sqrt(w1**2*e1[i]**2+(1.0-w1)**2*e2[i]**2))
        else:
            print('--- Error in the binary sum (wsum2).')
            print('--- The the weight of first set of data should be between 0 and 1.')
    else: # both weights are different of zero and are simply considered as factors
        for i in range(length):
            ysum.append(w1*y1[i]+w2*y2[i])
            esum.append(np.sqrt(w1**2*e1[i]**2+w2**2*e2[i]**2))
    ysum = np.array(ysum)
    esum = np.array(esum)
    x1 = x1.reshape(x1.shape[0],1)
    ysum = ysum.reshape(ysum.shape[0],1)
    esum = esum.reshape(esum.shape[0],1)
    summed = np.concatenate((x1,ysum),axis=1)
    summed = np.concatenate((summed,esum),axis=1)
    return summed
# End of wsum2
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def binary_sum(w1,data1,w2,data2):
    '''
    Sum (or subtract) two sets of values (y and error).
    Depending on the parameters w1 and w2, different operations are done.
    w1 and w2 are the weights for the sets 1 (y1,e1) and 2 (y2,e2).

    Case (1): both weights are zero (w1=w2=0)
        The function returns the sum weighted with the inverse of the square errors.
    Case (2): w1=0 and w2!=0
        The function returns the set 2 multiplied by w2
    Case (3): w1!=0 and w2=0
        If w1 is not in the range (0,1], error.
        Otherwise, the function returns the weighted sum with w2=1-w1
    Case (4): Both weights are different from zero (w1!=0 and w2!=0)
        The functions returns simply w1*y1+w2*y2.
        Note that the weights can also be negative, so this case includes the subtraction.

    Parameters
    ----------
    - w1, x1, y1, e1: weight and data for set 1
    - w2, x2, y2, e2: weight and data for set 2
        Data are given as 3 lists with abcissas, ordinates and errors

    Returns
    -------
    3 lists: floats
        The 3 lists are x, y and e
        - x: same abcissas as for the input data
        - y: summed ordinates
        - e: propagated errors

    Date: Fri Dec 22 2023
    Author: Gabriel Cuello
---------------------------------------
    '''

# Check that both input sets have the same shape
    if (data1.shape != data2.shape):
        print('--- Error in the binary sum (binary_sum).')
        print('--- The input data have not the same shape.')
        return None

# Make the weighted sum
    data3 = np.copy(data1)
    data3[:,1] = w1 * data1[:,1] + w2 * data2[:,1]
    data3[:,2] = np.sqrt(w1**2 * data1[:,2] + w2**2 * data2[:,2])

    return data3
# End of wsum2
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def get_xlim(xmin,xmax,dbin):
    '''
    This function calculates the limits and centers of bins for a histogram.

    Parameters
    ----------
    - xmin,xmax: float
        Minimum and maximum values of the binned variable
    - dbin: float
        Size of the binning box

    Returns
    -------
    - integer: number of bins in the binned range (nbins)
    - list: limiting values for each bin (dimension 1 x (nbins+1))
    - list: value corresponding to the center of each bin (dimension 1 x nbins).

    Called by:  rebin

    Created on Wed Dec 30, 2020
    Author: Gabriel Cuello
---------------------------------------
    '''
    xini = xmin-dbin/2.0 # coordinate of the left side of the first bin
    xfin = xmax+dbin/2.0 # coordinate of the right side of the last bin
    nb = int((xfin-xini)/dbin) # number of bins
    # Initialising the two output lists
#    x_lim = []
#    x_bin = []
    x_lim = [xmin - dbin / 2.0 + i * dbin for i in range(nb + 1)]
    x_bin = [xmin + i * dbin for i in range(nb)]
#    for i in range(nb+1):
#        x_lim.append(xmin-dbin/2.0+i*dbin)
#        x_bin.append(xmin+i*dbin)
#    x_bin.pop() # Removing the last element of this list
    return nb,x_lim,x_bin
# End of det_xlim
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def get_bins(x,xdel,x_lim):
    '''
    Determine the bins covered by a given point and the fraction of that point
    on each bin.

    A point is represented by a rectangle (in fact, it is part of a histogram).
    This is a general function, but in the particular case of counts on a
    detector, the center of the rectangle is the abcissa where the particle has
    been detected, and the width corresponds to the precision of the detection
    system, i.e., the space between 2 detection cells.

    Parameters
    ----------
    - x: float
        Abcissa of the experimental value
    - xdel: float
        Width covered by the experimental value
    - x_lim: list of floats
        List with the limits of each bin

    Returns
    -------
    - list: list of bins covered (completely or partially) by the rectangle
    - list: list containing the fraction of the rectangle covering each bin

    Both lists of floats have the same length.

    Called by:  rebin

    Date: Wed Dec 30, 2020
    Author: Gabriel Cuello
---------------------------------------
    '''
    # Initialising the two output lists
    bins = []
    frac = []
    # Determines the size of a bin, which is constant
    dbin = x_lim[1]-x_lim[0]

    # The rectangle corresponding to the x value has a width of xdel,
    # half on each side of x
    x1 = x-xdel/2.0 # coordinate of the left side  of the rectangle
    x2 = x+xdel/2.0 # coordinate of the right side of the rectangle

    # Determining the bins where left and right sides of the rectangle fall
    b1 = int((x1-x_lim[0])/dbin)
    b2 = int((x2-x_lim[0])/dbin)
    if b1 < 0 or b2 >= len(x_lim):
        return bins,frac
    deltab = b2-b1 # number of covered bins minus 1
    # There are 3 possible cases:
    #    (1) deltab = 0
    #        The rectangle completely falls on a single bin.
    #    (2) deltab = 1
    #        The rectangle falls on 2 bins, covering partially each one.
    #    (3) deltab > 1
    #        The rectangle falls on more than 2 bins, covering partially the
    #        first and the last ones, and completely the bins in between.
    if deltab == 0:         # Case (1)
    #   b1 (=b2) is the single bin where the rectangle falls.
    #   Then the fraction is equal to 1.0
        bins.append(b1)
        frac.append(1.0)
    elif deltab == 1:       # Case (2)
        f1 = (x_lim[b1+1]-x1)/xdel
        bins.append(b1)
        frac.append(f1)
        bins.append(b1+1)
        frac.append(1.0-f1)
    elif deltab > 1:        # Case (3)
        f1 = (x_lim[b1+1]-x1)/xdel # First bin
        bins.append(b1)
        frac.append(f1)
        for i in range(1,deltab): # Intermediate bins
            bins.append(b1+i)
            frac.append(dbin/xdel)
        f2 = (x2-x_lim[b2])/xdel # Last bin
        bins.append(b2)
        frac.append(f2)
    else:
        print ('ERROR in get_bins')

    return bins,frac
# End of get_bins
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def rebin(xdel,wlength,data,xmin,xmax,dbin):
    '''
    Make a rebinning of the experimetal data.

    Parameters
    ----------
    - xdel: float
        Witdh of a channel in scattering angle, assumed constant for the whole
        diffractogram. At D4, having 64 channels covering 8 degrees, this value
        is 0.125 degrees.
    - wlength: float
        Wavelength in Å. If positive, the binning is in Q-scale (1/Å).
        Otherwise, the binning in scattering angle (degrees).
    - data: matrix (ndata x 3), float
        col 0: abcissa in degrees if angular scale or in 1/Å if Q-scale
        col 1: intensity in arbitraty units
        col 2: intensity error in the same units as col 2

    Returns
    -------
    - matrix: float (nbins x 3)
        col 0: abcissa in degrees if angular scale or in 1/Å if Q-scale
        col 1: intensity in arbitraty units
        col 2: intensity error in the same units as col 2

    Requires:
        get_xlim, get_bins

    Date: Wed Dec 30, 2020
    Author: Gabriel Cuello
---------------------------------------
    '''
    x_dat = data[:,0]
    y_dat = data[:,1]
    e_dat = data[:,2]
    print ('0 ',len(x_dat),len(y_dat),len(e_dat))

    # Call get_xlim to obtain the number of bins, the limiting values for the bins and the values
    # of the bins
    nbins,x_lim,x_bin = get_xlim(xmin,xmax,dbin)

    # Creates lists for storing the new y, new error and the fraction
    y_bin = []
    e_bin = []
    f_bin = []

    # Initialise the lists that will serve as accumulators
    for i in range(nbins+1):
        y_bin.append(0.0)
        e_bin.append(0.0)
        f_bin.append(0.0)

    if (wlength <= 0):
        # The binning is in angular scale
        for i in range(len(x_dat)):
            if (np.isnan(y_dat[i]) == False) and (np.isnan(e_dat[i]) == False):
                # For each experimental x value, calls get_bins, which returns the bins covered by that
                # point and the corresponding fractions
                bins,frac = get_bins(x_dat[i],xdel,x_lim)
                # For each of these bins we add the corresponding fraction of y and error.
                # Because these fractions act as weighting factors, the fractions are accumulated for
                # normalisation purposes
                for j in range(len(bins)):
                    y_bin[bins[j]] += frac[j]*y_dat[i]
                    e_bin[bins[j]] += frac[j]*e_dat[i]
                    f_bin[bins[j]] += frac[j]
    else:
        # The binning is in Q-scale
        for i in range(len(x_dat)):
            if (np.isnan(y_dat[i]) == False) and (np.isnan(e_dat[i]) == False):
                # ${\rm d}Q = \frac{2\pi}{\lambda} \sqrt{1-\frac{Q\lambda}{4 \pi}}
                #  {\rm d}2\theta \frac{pi}{180}$
                qdel = 2.0*np.pi/wlength * np.sqrt(1.0-(x_dat[i]*wlength/4.0/np.pi)**2)
                qdel *= xdel * np.pi/180.0
                # For each experimental x value, calls get_bins, which returns the bins covered by that
                # point and the corresponding fractions
                bins,frac = get_bins(x_dat[i],qdel,x_lim)
                # For each of these bins we add the corresponding fraction of y and error.
                # Because these fractions act as weighting factors, the fractions are accumulated for
                # normalisation purposes
                for j in range(len(bins)):
                    y_bin[bins[j]] += frac[j]*y_dat[i]
                    e_bin[bins[j]] += frac[j]*e_dat[i]
                    f_bin[bins[j]] += frac[j]

    # Normalisation of y and errors. If fraction is 0, that bin has no data.
    for i in range(nbins+1):
        if (f_bin[i] != 0):
            y_bin[i] /= f_bin[i]
            e_bin[i] /= f_bin[i]
    x_bin = np.array(x_bin)
    y_bin = np.array(y_bin[0:-1])
    e_bin = np.array(e_bin[0:-1])
#    print ('1 ',len(x_bin),len(y_bin),len(e_bin))
    #   Reshapes the arrays as 2D arrays, but with 1 column
    x_bin=x_bin.reshape(x_bin.shape[0],1)
    y_bin=y_bin.reshape(y_bin.shape[0],1)
    e_bin=e_bin.reshape(e_bin.shape[0],1)
#    print ('2 ',len(x_bin),len(y_bin),len(e_bin))
    #   Concatenates the arrays to have a 3-column matrix
    binned = np.concatenate((x_bin,y_bin),axis=1)
    binned = np.concatenate((binned,e_bin),axis=1)
    return binned
# End of rebin
#--------1---------2---------3---------4---------5---------6---------7---------

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#
#  Physical Magnitudes
#
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

#--------1---------2---------3---------4---------5---------6---------7---------8
def getMassNumber(molarMass=6.0):
    """
    Calculate the mass number defined as:
        A = molar_mass / neutron_mass

    Parameters
    ----------
    - molarMass:  float
        the molar mas in amu. The default is 6.0 (carbon molar mass).

    Returns
    -------
    - float: the mass number

    Author: Gabriel Cuello
    Date: Jan 13, 2022
--------------------------------------------------------------------------------
    """
    neutronMass = 1.0086649 # in amu
    return molarMass / neutronMass
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def getFreeXS(BoundXS=5.08,A=50.9415):
    """
    Calculate the free cross section.

    The free cross section is calculated as:
        sigma_free = sigma_bound * A**2 /(1+A)**2
    where sigma_bound is the bound cross section (in barns) and A is the ratio
    mass/neutron_mass.

    Parameters
    ----------
    - BoundXS: float
        bound cross section in barns
        The default is 5.08 barns, the vanadium bound cross section.
    - A: float
        ratio mass/neutron_mass
        The default is 50.9415 amu, the vanadium mass number.

    Returns
    -------
    - float: the free cross section (same units as the bound cross section)

    Author: Gabriel Cuello
    Date: Jan 13, 2022
--------------------------------------------------------------------------------
    """
    return BoundXS * A**2 / (1.0+A)**2
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def getAbsXS(AbsXS=5.08,wavelength=1.8089):
    """
    Calculate the absorption cross section for a given wavelength.

    The absorption cross section is calculated as:
        sigma_abs = sigma_thermal * lambda / 1.8089 Å
    where lambda is the neutron wavelength in Å.

    Parameters
    ----------
    - AbsXS: float
        absorption cross section in barns for thermal neutrons
    - wavelength: float
        neutron wavelength
        The default is 1.8089Å.

    Returns
    -------
    - float: the absorption cross section for a given wavelength.

    Author: Gabriel Cuello
    Date: Dec 9, 2023
--------------------------------------------------------------------------------
    """
    return AbsXS * wavelength / 1.8089
#--------1---------2---------3---------4---------5---------6---------7---------8


#--------1---------2---------3---------4---------5---------6---------7---------8
def extractDict(dictionary,key):
    """
    Extract a sub-dictionary from a bigger one.

    This function extracts a transversal dictionary from a bigger dictionary
    containig dictionaries. The extracted dictionary will have the same keys
    that the bigger one, but the values will be only those corresponding to the
    input key.

    Parameters
    ----------
    - dictionary : dictionary
        The big dictionary containing sub-dictionaries.
    - key: string
        The key of the subdictionaries that will be extracted.

    Returns
    -------
    - dictionary: a dictionary with the keys of the big one and containing
    only the elements with the given key.

    Example
    -------
    For the following dictionary
        elements = {'Ti':{'CohXS': 1.485,'IncXS': 2.87,  'AbsXS':6.09 },
                    'Nb':{'CohXS': 6.253,'IncXS': 0.0024,'AbsXS':1.15 },
                    'P': {'CohXS': 3.307,'IncXS': 0.005, 'AbsXS':0.172}}
    The instruction: incohXS = extractDict(elements,'IncXS') will produce:
        incohXS = {'Ti':2.87,'Nb':0.0024,'P':0.005}

    Author: Gabriel Cuello
    Date: Oct 29, 2021
--------------------------------------------------------------------------------
    """
    result = {}
    for key1,value in dictionary.items():
        result[key1] = dictionary[key1][key]
    return result
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def extractAttr(Dict,attr):
    """
    Extract an attribue from a dictionary containing classes.

    Parameters
    ----------
    - Dict: dictionary
    - attr: attribute to be extracted

    Author: Jose Robledo
    Date: Sept 29, 2023
--------------------------------------------------------------------------------
    """
    result = {}
    for key, val in Dict.items():
        if hasattr(val, attr):
            result[key] = getattr(val, attr)
        else:
            print(f"Attribute {attr} not found in {val}.")
    return result
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def getAtomicDensity(density=1.0,molarMass=6.0):
    """
    Calculate the atomic density (atoms/volume).

    Given the macroscopic density (g/cm3) and the average atomic molar mass
    (g/mole), this function returns the atomic density (atoms/A3)

    Atomic density [atom/A3] =
           density [g/cm3] * NA [atom/mole] / molarM [g/mole] * 10**(-24)
    where molarM is the average molar mass per atom and NA = 6.02214076 10**23
    atoms/mole is the Avogadro's number.

    If density is less or equal to 0, the function returns water atomic density,
    i.e, 0.1 at/A3.

    Parameters
    ----------
    - density: float
        macroscopic density in g/cm3. The default is 1.0.
    - molarMass: float
        average molar mass per atom, in g/mole/atom. The default is 6.0.

    Returns
    -------
    - float: atomic density, in atoms/A3

    Example
    -------
    A water molecule has 3 atoms and the average molar mass is 18 g/mole/3 =
    6 g/mole, then
    AtomicDensity = 1 g/cm3 * 0.602214076 at/mole / 6 g/mole = 0.1 at/A3

    Author: Gabriel Cuello
    Date: Oct 25, 2021
--------------------------------------------------------------------------------
    """
# Avogadros's number times 10^(-24):
#    6.02214076 10^(23) * 10^(-24) = 0.602214076
# NA = 0.602214076
    if density <= 0:
        density = 1.0
        molarMass = 6.0
        print ('Attention!')
        print ('    Using the water density in getAtomicDensity function.')
    return density * 0.602214076 / molarMass
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def getDensity(atomic_density=0.1,molarM=6.0):
    """
    Calculate the macroscopic density (g/cm3).

    Given the atomic density (in atoms/A3) and the average atomic molar mass
    (g/mole), this function returns the macroscopic density (g/cm3) .

    density [g/cm3] =
        Atomic density [atoms/A3] / NA [atoms/mole] * molarM [g/mole] * 10**24
    where molarM is the average molar mass per atom and
    NA = 6.02214076 10**23 atoms/mole is the Avogadro's number.

    If density is less or equal to 0, the function returns water density,
    1.0 g/cm3

    Parameters
    ----------
    -  atomic density: float
        atomic density in atoms/A3. The default is 0.1 atoms/A3.
    -  molarMass: float
        average molar mass per atom, in g/mole/atom. The default is 6.0.

    Returns
    -------
    -  float: macroscopic density, in g/cm3

    Example
    -------
    A water molecule has 3 atoms and the average molar mass is
    18 g/mole/3 = 6 g/mole/atom
        Density = 0.1 atoms/A3 / 0.602214076 atoms/mole * 6 = 1 g/cm3

    Author: Gabriel Cuello
    Date: Oct 25 2021
--------------------------------------------------------------------------------
    """
    # Avogadros's number times 10^(-24):
    # 6.02214076 10^(23) * 10^(-24) = 0.602214076
    # NA = 0.602214076
    result = atomic_density / 0.602214076 * molarM
    if atomic_density <= 0:
        result = 1.0
        print ('Attention! Using the water density in getDensity function.')
#    print('Density ',"{:.6f}".format(result),' g/cm3')
    return result
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def getNofAtoms(atoms):
    """
    Calculate the number of atoms in the basic unit.

    Parameters
    ----------
    - atoms: dictionary
        This dictionary contains the number of atoms in the sample.
        The keys are the chemical symbols and values the corresponding number
        of atoms in the sample

    Returns
    -------
    - float: the number of atoms in a basic unit (a kind of molecule)

    Example
    -------
    If atoms = {'H': 2, 'O': 1}, representing a water molecule, the output
    should be 3.

    Date: Sat Oct 23 2021
    Author: Gabriel Cuello
---------------------------------------
    """
    natoms = 0
    for key,value in atoms.items():
        natoms += value
    return natoms
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def getConcentrations(atoms):
    """
    Calculate the concentration of each atom in the sample.

    Parameters
    ----------
    - atoms: dictionary
        This dictionary contains the number of atoms in the sample.
        The keys are the chemical symbols and values the corresponding number
        of atoms in the sample

    Returns
    -------
    - dictionary: with the same keys as the input dictionary and the
      corresponding concentration values.

    Requires
    --------
    - getNofAtoms

    Example
    -------
    If atoms = {'H': 2, 'O': 1}, representing a water molecule,
    the output should be {'H': 0.666666, 'O': 0.333333}

    Date: Sat Oct 23 2021
    Author: Gabriel Cuello
---------------------------------------
    """
    concentration = {}
    natoms = sum(atoms.values())
#    natoms = getNofAtoms(atoms)
    if natoms <= 0:
        print ('---> ERROR! The number of atoms must be positive.')
#    print ('Atomic concentrations:')
    for key, value in atoms.items():
        concentration[key] = value/natoms
#        print ('   ',key,'=',"{:.6f}".format(concentration[key]))
    return concentration
#--------1---------2---------3---------4---------5---------6---------7---------

def AtomicAvg(concentration,magnitude):
    """
    Make an atomic average of a given magnitude.

    Parameters
    ----------
    - concentration: dictionary
        a dictionary with the concentration of each atom in the sample
    - magnitude: dictionary
        a dictionary with the magnitude to average

    Note that both dictionaries must have the same keys, i.e.,
    the chemical symbols

    Returns
    -------
    - float: the average of the given magnitude

    Date: Wed Dec 30 2020
    Author: Gabriel Cuello
---------------------------------------
    """
    average = 0
    for key,value in concentration.items():
        average += float(value) * float(magnitude[key])
    return average
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def XS_model(Eval, Eeff, composition_vec, bound_xs_vec, A_vec):
    """
    Calculate the neutron total cross section in the epithermal limit.

    $\sigma = \sum_i n_i \sigma_{b,i} \left(\frac{A_i}{A_i+1}\right)**2
    \left( 1 + \frac{Eff}{2 A_i Eval}\right$

    Note that the input lists must be consistenly ordered.

    Parameters
    ----------
    Eval : float
        Energy value, in meV
    Eeff : float
        Effective Energy = kB * Teff, in meV
    composition_vec : list
        ordered list of composition coefficients.
    bound_xs_vec : list
        ordered list of bound cross sections (barns)
    A_vec : list
        ordered list of molar mass numbers (amu)

    Returns
    -------
    float: Total cross section for the given energy.

    Date Wed Feb 23, 2022
    Author: José Robledo
 ---------------------------------------
   """
    composition_vec = np.array(composition_vec)
    bound_xs_vec = np.array(bound_xs_vec)
    A_vec = np.array(A_vec)

    a_vals = (A_vec/(A_vec + 1))**2

    result = np.sum(composition_vec * bound_xs_vec * a_vals * (1 + Eeff / (2 * A_vec * Eval)))
    return result
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def scattering_probability(E0, scat_xs, composition_vec, abs_xs_vec):
    """
    Calculate the scattering probability for a given neutron incident energy.

    Parameters
    ----------
    E0 : float
        Incident energy in meV.
    bound_xs : float
        Scattering cross section
    composition_vec : list
        ordered list with compositions.
    abs_xs_vec : list
        ordered list with absorption scattering cross sections.

    Returns
    -------
    - float: Scattering probability for a neutron of energy E0.

    Date: Wed Feb 23, 2022
    Author: José Robledo
 ---------------------------------------
    """
    composition_vec = np.array(composition_vec)
    abs_xs_vec = np.array(abs_xs_vec)

    # calculate total absorption cross section
    abs_xs = np.dot(composition_vec, abs_xs_vec) / np.sqrt(E0/25.3)

    # calculate total cross section
    total_xs = abs_xs + scat_xs

    return 1 - abs_xs/total_xs
#--------1---------2---------3---------4---------5---------6---------7---------

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#
#  Experimental Settings
#
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

#--------1---------2---------3---------4---------5---------6---------7---------8
def setExpInfo(Proposal="6-01-000",mainProposer="nobody",
               experimenters="nobody",LC="Cuello",otherUsers="nobody",
               startDate="01/01/2000",endDate="02/01/2000",
               environment="A",logBook=0,logPage=0,instrument="D4"):
    """
    Create a dictionary with information about the experiment.

    The function also print out a summary on the screen.

    Parameters
    ----------
    Proposal : string, optional
        Proposal number. The default is "6-01-000".
    mainProposer : string, optional
        Main proposer. The default is "nobody".
    experimenters : string, optional
        Experiments on site. The default is "nobody".
    LC : string, optional
        The local contact. The default is "Cuello".
    otherUsers : string, optional
        Users in the proposal but no on site. The default is "nobody".
    startDate : string, optional
        Date of the starting day. The default is "01/01/2000".
    endDate : string, optional
        Date of the ending day. The default is "02/01/2000".
    environment : string, optional
        Sample environment. The default is "A".
    logBook : integer, optional
        Number of the D4 logbook. The default is 0.
    logPage : inetger, optional
        Page in the D4 logbook. The default is 0.
    instrument : string, optional
        Instrument. The default is "D4".

    Returns
    -------
    experiment : dictionary
        A dictionary containing the information about the experiment.

    Date: Sun Dec 26 2021
    Author: Gabriel Cuello
--------------------------------------------------------------------------------
    """
    experiment = {'Proposal': Proposal}
    experiment['MP'] = mainProposer
    experiment['experimenters'] = experimenters
    experiment['LC'] = LC
    experiment['otherUsers'] = otherUsers
    experiment['instr'] = instrument
    experiment['startDate'] = startDate
    experiment['endDate'] = endDate
    experiment['environment'] = environment
    experiment['logBook'] = logBook
    experiment['logPage'] = logPage
    print (30*'-')
    print ('Experiment')
    print (4*' ','Instrument: {}'.format(experiment['instr']))
    print (4*' ','Proposal: {}'.format(experiment['Proposal']))
    print (4*' ','Main proposer: {}'.format(experiment['MP']))
    print (4*' ','Other users: {}'.format(experiment['otherUsers']))
    print (4*' ','On-site users: {}'.format(experiment['experimenters']))
    print (4*' ','Local contact: {}'.format(experiment['LC']))
    print (4*' ','Starting date: {0} ---->  Ending date: {1}'.
           format(experiment['startDate'],experiment['endDate']))
    print (4*' ','Sample environment: {}'.format(experiment['environment']))
    print (4*' ','D4 notebook: {0}  Page: {1}'.
           format(experiment['logBook'],experiment['logPage']))
    print ()
    return experiment
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def setBeamInfo(zeroAngle=0,wavelength=0.5,
                LohenSlit=2.5,GamsSlit=-2.5,topFlag=25.0,bottomFlag=-25.0):
    """
    Create a dictionary with information about the beam.

    The function also print out a summary on the screen.

    Parameters
    ----------
    - zeroAngle : float, optional
        The zero-angle correction, as obtained from the calibration with Ni
        powder. The default is 0 deg.
    - wavelength : float, optional
        The incident wavelength, as obtained from the calibration with Ni
        powder. The default is 0.5 A.
    - LohenSlit : float, optional
        The position of the vertical slit (left in downstream direction).
        This is the slit at Lohengrin side. The default is 2.5 mm.
    - GamsSlit : float, optional
        The position of the vertical slit (right in downstream direction).
        This is the slit at GAMS side. The default is -2.5 mm.
    - topFlag : float, optional
        The position of the top horizontal slit. The default is 25.0 mm.
    - bottomFlag : float, optional
        The position of the top horizontal slit. The default is -25.0 mm.

    Returns
    -------
    - dictionary: A dictionary containing the information about the beam.

    Date: Sun Dec 26 2021
    Author: Gabriel Cuello
    ---------------------------------------
    """
    beam = {'zero': zeroAngle}    # zero angle correction
    beam['wlength'] = wavelength   # neutron wavelength
    # Vertical slits defining the horizontal size (width) of the beam
    # Lohengrin side, i.e., on the left in downstream direction
    # Gams side, i.e., on the right in downstream direction
    beam['LohenSlit'] = LohenSlit   # in mm
    beam['GamsSlit'] = GamsSlit     # in mm
    beam['width'] = beam['LohenSlit'] - beam['GamsSlit'] # Beam width (in mm)
    # Horizontal flags defining the vertical size (height) of the beam
    beam['topFlag'] = topFlag       # in mm
    beam['botFlag'] = bottomFlag    # in mm
    beam['height'] = beam['topFlag']-beam['botFlag'] # Beam height (in mm)
    print (30*'-')
    print ('Beam characteristics')
    print (4*' ','Wavelength = {:8.6g} A'.format(beam['wlength']))
    print (4*' ','Zero angle = {:8.6g} deg'.format(beam['zero']))
    print (4*' ','Dimensions: (Width = {0:.6g} mm)x(Height = {1:.6g} mm)'.
           format(beam['width'],beam['height']))
    print ()
    return beam
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def setCanInfo(material='Vanadium',shape='Cylinder',
               outerDiam=5,innerDiam=4.8,height=60.0):
    """
    Create a dictionary with information about the container.

    The function also print out a summary on the screen.

    Parameters
    ----------
    - material : string, optional
        Material of the container. The default is Vanadium.
    - shape : string, optional
        Shape of the container. The default is Cylinder.
    - outerDiam : float, optional
        Outer diameter of the container. The default is 5 mm.
    - innerDiam : float, optional
        Inner diameter of the container. The default is 4.8 mm.
    - height : float, optional
        Height of the container. The default is 60.0 mm.

    Returns
    -------
    - dictionary: A dictionary containing the information about the container.

    Date: Sun Dec 26 2021
    Author: Gabriel Cuello
    ---------------------------------------
    """
    can = {'material': material}
    can['shape'] = shape
    can['outerDiam'] = outerDiam   # outer diameter (in mm)
    can['innerDiam'] = innerDiam   # inner diameter (in mm)
    can['height'] = height     # height (in mm)
    can['wallThickness'] = (can['outerDiam']-can['innerDiam'])/2.0
    print (30*'-')
    print ('Container')
    print (4*' ','Type: {0} {1}'.format(can['material'],can['shape']))
    print (4*' ','Outer diameter = {} mm'.format(can['outerDiam']))
    print (4*' ','Inner diameter = {} mm'.format(can['innerDiam']))
    print (4*' ','Wall thickness = {:.3g} mm'.format(can['wallThickness']))
    print (4*' ','Height = {} mm'.format(can['height']))
    print ()
    return can
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def setBinInfo(AngularResolution=0.125,
               AMin=0.0, AMax=140.0, AStep=0.125,
               QMin=0.0, QMax=23.5,  QStep=0.02,
               RMin=0.0, RMax=20.0,  RStep=0.01):
    """
    Create a dictionary with information about the binnings.

    The function also print out a summary on the screen.

    Parameters
    ----------
    - AngularResolution : string, optional
        The angular space between 2 detection cells. The default 0.125 degrees.
        This value could be 0.0625 degrees whe the electronics will change to
        128 cells instead of the current 64 cells.
    - Min, Max, Step : 3 floats for each scale
        Minimum, maximum and step values for the binning of 3 scales.
        The 3 scales are angle (A), momentum (Q) and distance (R), for units of
        degrees, 1/Å and Å, respectively.

    Returns
    -------
    - dictionary: A dictionary containing the information about the binning.

    Date: Sun Dec 26 2021
    Author: Gabriel Cuello
    ---------------------------------------
    """
    # Experimental width of an angular channel (in degrees)
    # 0.125 deg (January 2022)
    binning = {'Ares': AngularResolution}
    # Tuples with initial, final and step for each scale
    #   in degrees for angular scale
    #   in 1/A for Q-scale
    #   in A for R-scale
    binning['Abin'] = (AMin,AMax,AStep)
    binning['Qbin'] = (QMin,QMax,QStep)
    binning['Rbin'] = (RMin,RMax,RStep)

    binning['NbrPointsA'] = int((binning['Abin'][1]-binning['Abin'][0])
                                /binning['Abin'][2])
    binning['NbrPointsQ'] = int((binning['Qbin'][1]-binning['Qbin'][0])
                                /binning['Qbin'][2])
    binning['NbrPointsR'] = int((binning['Rbin'][1]-binning['Rbin'][0])
                                /binning['Rbin'][2])
    print (30*'-')
    print ('Binning')
    print (4*' ','Angular channel width = {:.3g} deg'.format(binning['Ares']))
    print (4*' ','In angle: from {0:.3g} deg to {1:.3g} deg, in steps of {2:.3f} deg, thus {3:.5g} points.'.
           format(binning['Abin'][0],binning['Abin'][1],binning['Abin'][2],binning['NbrPointsA']))
    print (4*' ','In Q: from {0:.3g} 1/A to {1:.3g} 1/A, in steps of {2:.3f} 1/A, thus {3:.5g} points.'.
           format(binning['Qbin'][0],binning['Qbin'][1],binning['Qbin'][2],binning['NbrPointsQ']))
    print (4*' ','In R: from {0:.3g} A to {1:.3g} A, in steps of {2:.3f} A, thus {3:.5g} points.'.
           format(binning['Rbin'][0],binning['Rbin'][1],binning['Rbin'][2],binning['NbrPointsR']))
    print ()
    return binning
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def setNumorInfo(totalNumors=('Exp',0,1), containerNumors=('Can',0,1),
            environmentNumors=('Env',0,1),
            nickelNumors=('Ni5',0,1), vanadiumNumors=('Van',0,1),
            absorberNumors=('Abs',0,1), sampleNumors =('S01',0,1)):
    """
    Create a dictionary with information about the numors.

    This function is not very useful in its present status (dec 2023).
    Note that only one sample is accepted in the output dictionary.

    The function also print out a summary on the screen.

    Parameters
    ----------
    - totalNumors:       tuple of 3 elements, a string, first and last numor
    - containerNumors:   tuple of 3 elements, a string, first and last numor
    - environmentNumors: tuple of 3 elements, a string, first and last numor
    - nickelNumors:      tuple of 3 elements, a string, first and last numor
    - vanadiumNumors:    tuple of 3 elements, a string, first and last numor
    - absorberNumors:    tuple of 3 elements, a string, first and last numor
    - sampleNumors:      tuple of 3 elements, a string, first and last numor

    Returns
    -------
    - dictionary: A dictionary containing the information about the numors.

    Date: Sun Dec 26 2021
    Author: Gabriel Cuello
    ---------------------------------------
    """
    numors = {'experiment': totalNumors,'container': containerNumors,
              'environment': environmentNumors,
              'nickel': nickelNumors,'vanadium': vanadiumNumors,'absorber': absorberNumors,
              'sample': sampleNumors}

    print ('Numors')

    if totalNumors[1]!=0:
        print ('{}Total {}: {} -{}, {} numors.'.format(
            4*' ',*numors['experiment'],int(numors['experiment'][2])-int(numors['experiment'][1])+1))
    if containerNumors[1]!=0:
        print ('{}{}: {} -{}, {} numors.'.format(
            4*' ',*numors['container'],int(numors['container'][2])-int(numors['container'][1])+1))
    if environmentNumors[1]!=0:
        print ('{}{}: {} -{}, {} numors.'.format(
            4*' ',*numors['environment'],int(numors['environment'][2])-int(numors['environment'][1])+1))
    if nickelNumors[1]!=0:
        print ('{}{}: {} -{}, {} numors.'.format(
            4*' ',*numors['nickel'],int(numors['nickel'][2])-int(numors['nickel'][1])+1))
    if vanadiumNumors[1]!=0:
        print ('{}{}: {} -{}, {} numors.'.format(
            4*' ',*numors['vanadium'],int(numors['vanadium'][2])-int(numors['vanadium'][1])+1))
    if absorberNumors[1]!=0:
        print ('{}{}: {} -{}, {} numors.'.format(
            4*' ',*numors['absorber'],int(numors['absorber'][2])-int(numors['absorber'][1])+1))
    if sampleNumors[1]!=0:
        print ('{}Sample {}: {} -{}, {} numors.'.format(
            4*' ',*numors['sample'],int(numors['sample'][2])-int(numors['sample'][1])+1))
    return numors
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def setVanaInfo(IncXS=5.08,CohXS=0.0184,ScaXS=5.0984,AbsXS=5.08,
                CohSL=-0.3824,molarM=50.9415,NAtoms=1.0,
                Diam=6.08,Height=50.0,density=6.51):
    '''
    Create a dictionary with information about the vanadium.

    The function also print out a summary on the screen.

    Parameters
    ----------
    - IncXS: float
        Incoherent cross section, in barns. Default value = 5.08 barns
    - CohXS: float
        Coherent cross section, in barns. Default value = 0.0184 barns
    - ScaXS: float
        Scattering cross section, in barns. Default value = 5.0984 barns
    - AbsXS: float
        Absorption cross section, in barns. Default value = 5.08 barns
    - CohSL: float
        Coherent scattering length, in fm. Default value = -0.3824 fm
    - molarM: float
        Molar mass, in amu. Default value = 50.9415 amu
    - NAtoms: float
        Number of atoms in a basic unit. Because it is a monoatomic system, NAtoms = 1
    - Diam: float
        Diameter of the vanadium rod, in mm. Default value = 6.08 mm
    - Height: float
        Height of the vanadium rod in the beam, in mm. Default value = 50 mm
    - Density: float
        Density of vanadium, in g/cm3. Default value = 6.51 g/cm3

    Returns
    -------
    - dictionary: A dictionary containing the information about the vanadium.

    Date: Sun Dec 26 2021
    Author: Gabriel Cuello
    ---------------------------------------
    '''
    vana = {}
    vana['IncXS'] = IncXS # Incoherent scattering cross section (sigma bound), in barns
    vana['CohXS'] = CohXS # Coherent scattering cross section, in barns
    vana['ScaXS'] = ScaXS # Coherent scattering cross section, in barns
    vana['AbsXS'] = AbsXS # Absorption cross section, in barns
    vana['CohSL'] = CohSL # Coherent scattering length, in fm
    vana['SelfQ0'] = vana['IncXS']/4./np.pi
    vana['diam'] = Diam # diameter (mm)
    vana['NAtoms'] = NAtoms # Number of atoms in a unit
    vana['molarM'] = molarM # Molar mass of vanadium, in g/mol
    vana['den_gcc'] =density # Macroscopic density, in g/cm3
    vana['den_aac'] = getAtomicDensity(density=vana['den_gcc'],molarMass=vana['molarM'])
    vana['A'] = getMassNumber(molarMass=vana['molarM']) # Ratio mass to neutron mass
    vana['FreeIncXS'] = getFreeXS(BoundXS=vana['IncXS'],A=vana['A'])
    vana['volume'] = getCylVolume(diameter=vana['diam'],height=Height)/1000.0
    print (30*'-')
    print ('Standard of vanadium')
    print(4*' ','The standard is a cylinder of {:.3g} mm of diameter.'.format(vana['diam']))
    print(4*' ','Volume in the beam = {:.3g} cm3.'.format(vana['volume']))
    print ()
    print(4*' ','Bound incoherent cross section {:.6g} barns/atom.'.format(vana['IncXS']))
    print(4*' ','Free incoherent cross section',"{:.6g} barns/atom.".format(vana['FreeIncXS']))
    print ()
    print(4*' ','b**2 = sigma/4/pi at Q=0 (self) is',"{:.6g} barns/sterad/atom.".format(vana['SelfQ0']))
    print(4*' ','b**2 = sigma/4/pi at Q=infty is',"{:.6g} barns/sterad/atom".format(vana['FreeIncXS']/4./np.pi))
    print ()
    print(4*' ','The molar mass is',"{:.6g} g/mol.".format(vana['molarM']))
    print(4*' ','Mass number, A = ',"{:.6g} (= mass/neutron_mass)".format(vana['A']))
    print ()
    print(4*' ','Density = ',"{:.6g} g/cm3.".format(vana['den_gcc']))
    print(4*' ','Atomic density = {:.6g} atoms/A3.'.format(vana['den_aac']))
    print ()
    return vana
#--------1---------2---------3---------4---------5---------6---------7---------


#--------1---------2---------3---------4---------5---------6---------7---------
def setSampleInfo(comp_sample,atoms_sample,natoms_sample,
                          elements,c_sample,wavelength=0.5,beamH=50.0,
                          vanavol=1.0,vanadens=0.0769591,
                          height=60.0,diameter=5.0,mass=1.0,density=1.0,
                          title='sample'):
    '''
    Create a dictionary with information about the sample.

    The function also print out a summary on the screen.

    Parameters
    ----------
    - comp_sample:

    - atoms_sample:

    - natoms_sample:

    - elements:

    - c_sample:

    - wavelength: float
        Wavelength, in Å. Default value = 0.5 Å
    - beamH: float
        Height of the beam, in mm. Default value = 50.0 mm.
    - vanavol: float
        Volume of vanadium rod, in cm3. Default value = 1.0 cm3
    - height: float
        Height of the sample, in mm. Default value = 60 mm
    - diameter: float
        Diameter of the sample, in mm. Default value = 5.0 mm
    - mass: float
        Mass of the sample, in g. Default value = 1.0 g
    - density: float
        Density of the sample, in g/cm3. Default value = 1.0 g/cm3
    - title: string
        Title to identify the sample. Default value = 'sample'

    Returns
    -------
    - dictionary: A dictionary containing the information about the sample.

    Date: Sun Dec 26 2021
    Author: Gabriel Cuello
    ---------------------------------------
    '''
    # Creating a new dictionary with the sample basic information
    sample ={'Title':title}

    sample['height'] = height # sample height (in mm)
    # This is the height of the sample seen by the beam
    sample['heightInBeam'] = min(sample['height'],beamH)
    # sample diameter, i.e., container inner diameter
    # If it is a delf-contained sample (no container), put the sample diameter (in mm)
    sample['diam'] = diameter
    sample['mass'] = mass # sample mass (in g)
    sample['NAtoms'] = natoms_sample # number of atoms in 1 unit

    # average molar mass per atom (in g/mol/atom)
    sample['molarM'] = AtomicAvg(c_sample,extractAttr(elements,'weight'))

    # Volume = pi*radius^2*height (in cm3)
    sample['volume'] = getCylVolume(sample['diam'],sample['height'])/1000.0
    # Volume of the sample in the beam (in cm3)
    sample['volumeInBeam'] = getCylVolume(diameter=sample['diam'],height=sample['heightInBeam'])/1000.0

    sample['den_gcc'] = density # macroscopic density of the sample (in g/cm3)

    # Effective density, simply mass/volume (g/cm3)
    sample['effden_gcc'] = sample['mass']/sample['volume']

    # Packing fraction
    sample['packing'] = sample['effden_gcc']/sample['den_gcc']
    # Filling fraction:
    #    1 if completely-filled container, less than 1 for a partially-filled container
    sample['filling'] = sample['heightInBeam']/beamH

    # atomic density (atoms/A3)
    sample['den_aac'] = getAtomicDensity(density=sample['den_gcc'],molarMass=sample['molarM'])

    # Effective atomic density (atoms/A3)
    sample['effden_aac'] = getAtomicDensity(density=sample['effden_gcc'],molarMass=sample['molarM'])

    # Coherent scattering cross section (barns)
    sample['CohXS'] = AtomicAvg(c_sample,extractAttr(elements,'sig_coh'))
    # Incoherent scattering cross section (barns)
    sample['IncXS'] = AtomicAvg(c_sample,extractAttr(elements,'sig_inc'))
    # Scattering cross section (barns)
    sample['ScaXS'] = AtomicAvg(c_sample,extractAttr(elements,'sig_sca'))
    # Absorption cross section (barns)
    sample['AbsXS'] = AtomicAvg(c_sample,extractAttr(elements,'sig_abs'))
    # Absorption cross section at working wavelength (barns)
    sample['AbsWW'] = sample['AbsXS'] * wavelength/1.8
    # Coherent scattering length (fm)
    sample['CohSL'] = AtomicAvg(c_sample,extractAttr(elements,'re_bcoh'))
    # Atomic number (amu)
    sample['A'] = AtomicAvg(c_sample,extractAttr(elements,'A'))
    # Free coherent scattering cross section (barns)
    sample['FreeCohXS'] = getFreeXS(BoundXS=sample['CohXS'],A=sample['A'])
    # Free incoherent scattering cross section (barns)
    sample['FreeIncXS'] = getFreeXS(BoundXS=sample['IncXS'],A=sample['A'])
    # Free scattering cross section (barns)
    sample['FreeScaXS'] = getFreeXS(BoundXS=sample['ScaXS'],A=sample['A'])
    # Self incoherent scattering cross section at Q=0 (barns), selfQ0 = IncXS/4/pi
    sample['SelfQ0'] = sample['IncXS']/4.0/np.pi

    # Ratio of sample volume to vanadium volume
    sample['VolRatioSV'] = sample['volumeInBeam']/vanavol
    # Ratio of sample density to vanadium desity
    sample['DenRatioSV'] = sample['effden_aac']/vanadens

    print (80*'-')
    print ('Sample',sample['Title'])

    text = ' '
    print ()
    print ('Composition:')
    for key,value in comp_sample.items():
        text += '+ '+str(value)+' of '+key+' '
    #    print (8*' ',key,value)
    print (4*' ','A "unit" is',text[3:])

    text = ' '
    print ()
    print ('Atomic composition:')
    for key,value in atoms_sample.items():
        text += '+ '+str(value)+' '+key+' atoms '
    #    print (8*' ',key,value)
    print (4*' ','A "unit" has',text[3:])
    print (4*' ','Number of atoms in one unit = {} atoms'.format(sample['NAtoms']))


    print ()
    print ('Atomic concentrations:')
    for key,value in c_sample.items():
        print (8*' ',key,"{:.6f}".format(value))

    print ()
    print ('Average molar mass: {:.6g} g/mol/atom'.format(sample['molarM']))
    print ('Mass number: {:.6g} amu'.format(sample['A']))
    print ()
    print ('Coherent cross section: {:.6g} barns/atom'.format(sample['CohXS']))
    print ('Incoherent cross section: {:.6g} barns/atom'.format(sample['IncXS']))
    print ('Scattering cross section: {:.6g} barns/atom'.format(sample['ScaXS']))
    print ()
    print ('Free coherent cross section: {:.6g} barns/atom'.format(sample['FreeCohXS']))
    print ('Free incoherent cross section: {:.6g} barns/atom'.format(sample['FreeIncXS']))
    print ('Free scattering cross section: {:.6g} barns/atom'.format(sample['FreeScaXS']))
    print ()
    print ('Absorption cross section: {:.6g} barns/atom'.format(sample['AbsXS']))
    print ('Absorption cross section at {:.6g} A: {:.6g} barns/atom'.format(wavelength,sample['AbsWW']))
    print ()
    print ('Coherent scattering length: {:.6g} fm'.format(sample['CohSL']))
    print ('Self scattering cross section at Q=0: {:.6g} barns/steradian/atom'.format(sample['SelfQ0']))
    print ()
    print ('Density: {:.6g} g/cm3'.format(sample['den_gcc']))
    print ('Atomic density: {:.6g} atoms/A3'.format(sample['den_aac']))
    print ()
    print ('Cylindrical sample')
    print (4*' ','Diameter: {:.6g} mm'.format(sample['diam']))
    print (4*' ','Height: {:.6g} mm'.format(sample['height']))
    print (4*' ','Volume: {:.6g} cm3'.format(sample['volume']))
    print (4*' ','Mass: {:.6g} g'.format(sample['mass']))
    print (4*' ','Effective density: {:.6g} g/cm3'.format(sample['effden_gcc']))
    print (4*' ','Effective atomic density: {:.6g} atoms/A3'.format(sample['effden_aac']))
    print ()
    print (4*' ','Sample/Vanadium volume fraction: {:.6g}'.format(sample['VolRatioSV']))
    print (4*' ','Sample/Vanadium density fraction: {:.6g}'.format(sample['DenRatioSV']))
    print (4*' ','Packing fraction: {:.6g}'.format(sample['packing']))
    print (4*' ','Filling fraction: {:.6g}'.format(sample['filling']))

    return sample
#--------1---------2---------3---------4---------5---------6---------7---------

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#
#  Fitting Models
#
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

#--------1---------2---------3---------4---------5---------6---------7---------
def fittingRange0(xmin,xmax,ymin,ymax,abcissas,ordinates,errors):
    '''
    Define a subset of a set of data. The subset is the number of points that
    are inside a rectangle.

    Parameter
    ---------
    - xmin, xmax: 2 floats
        Minimum and maximum values on the abcissa axis
    - ymin, ymax: 2 floats
        Minimum and maximum values on the ordinate axis
    - abcissas: array of floats
        An array with the abcissas axis
    - ordinates: array of floats
        An array with the ordinates axis
    - errors: array of floats
        An array with the errors

    Returns
    -------
    - matrix: A 3-column matrix containing the abcissas, ordinates and errors
        included in the rectangle xmix < x < xmax, ymin < y < ymax

    Date: Sun Dec 26 2021
    Author: Gabriel Cuello
    ---------------------------------------
    '''
    if (len(abcissas) != len(ordinates)):
        print ('ERROR: Ordinate and abcissa must have the same length in function fittingRange.')
    if (xmax<xmin) or (ymax<ymin):
        print ('ERROR: The limits of the rectangle are wrong in the function fittingRange.')
    x_range = []
    y_range = []
    e_range = []
    for i in range(len(abcissas)):
        if (abcissas[i] >= xmin) and (abcissas[i] <= xmax):
            if (ordinates[i] >= ymin) and (ordinates[i] <= ymax):
                x_range.append(abcissas[i])
                y_range.append(ordinates[i])
                e_range.append(errors[i])
    # np.array converts a list in a numpy array
    if (len(x_range)<10):
        print ('WARNING: Less than 10 points for the fitting in fittingRange.')
    if (len(x_range)==0):
        print ('ERROR: No points for the fitting in fittingRange.')

    x = np.array(x_range)
    y = np.array(y_range)
    e = np.array(e_range)
    x = x.reshape(x.shape[0],1)
    y = y.reshape(y.shape[0],1)
    e = e.reshape(e.shape[0],1)
    limited = np.concatenate((x,y),axis=1)
    limited = np.concatenate((limited,e),axis=1)
    return limited
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def fittingRange(xmin,xmax,ymin,ymax,abcissas,ordinates,errors):
    '''
    Define a subset of a set of data. The subset is the number of points that
    are inside a rectangle.

    Parameter
    ---------
    - xmin, xmax: 2 floats
        Minimum and maximum values on the abcissa axis
    - ymin, ymax: 2 floats
        Minimum and maximum values on the ordinate axis
    - abcissas: array of floats
        An array with the abcissas axis
    - ordinates: array of floats
        An array with the ordinates axis
    - errors: array of floats
        An array with the errors

    Returns
    -------
    - 3 arrays: Three arrays of floats containing the abcissas, ordinates and errors
        included in the rectangle xmix < x < xmax, ymin < y < ymax

    Date: Sun Dec 26 2021
    Author: Gabriel Cuello
    ---------------------------------------
    '''
    if (len(abcissas) != len(ordinates)):
        print ('ERROR: Ordinate and abcissa must have the same length in function fittingRange.')
    if (xmax<xmin) or (ymax<ymin):
        print ('ERROR: The limits of the rectangle are wrong in the function fittingRange.')
    x_range = []
    y_range = []
    e_range = []
    for i in range(len(abcissas)):
        if (abcissas[i] >= xmin) and (abcissas[i] <= xmax):
            if (ordinates[i] >= ymin) and (ordinates[i] <= ymax):
                x_range.append(abcissas[i])
                y_range.append(ordinates[i])
                e_range.append(errors[i])
    # np.array converts a list in a numpy array
    if (len(x_range)<10):
        print ('WARNING: Less than 10 points for the fitting in fittingRange.')
    if (len(x_range)==0):
        print ('ERROR: No points for the fitting in fittingRange.')

    x = np.array(x_range)
    y = np.array(y_range)
    e = np.array(e_range)
    x = x.reshape(x.shape[0],1)
    y = y.reshape(y.shape[0],1)
    e = e.reshape(e.shape[0],1)
#    limited = np.concatenate((x,y),axis=1)
#    limited = np.concatenate((limited,e),axis=1)
    return x,y,e
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def fit_range(xmin,xmax,ymin,ymax,data):
    """
    Define a rectangle in the x,y plane. All points inside this rectangle will
    be used in the fitting procedure.

    Parameter
    ---------
    - xmin, xmax: 2 floats
        Minimum and maximum values on the abcissa axis
    - ymin, ymax: 2 floats
        Minimum and maximum values on the ordinate axis
    - data: matrix of floats
        Three column matrix containing abcissas, ordinates, errors

    Returns
    -------
    - matrix: A 3-column matrix containing the abcissas, ordinates and errors
        included in the rectangle xmix < x < xmax, ymin < y < ymax

    Date: Thu Jan 01 2021
    Author: Gabriel Cuello
---------------------------------------
    """
    x_dat = data[:,0]
    y_dat = data[:,1]
    e_dat = data[:,2]
    if (len(x_dat) != len(y_dat)):
        print ('ERROR: Ordinate and abcissa must have the same length in the function fit_range.')
    if (xmax<xmin) or (ymax<ymin):
        print ('ERROR: The limits of the rectangle are wrong in the function fit_range.')
    x_range = []
    y_range = []
    e_range = []
    for i in range(len(x_dat)):
        if (x_dat[i] >= xmin) and (x_dat[i] <= xmax):
            if (y_dat[i] >= ymin) and (y_dat[i] <= ymax):
                x_range.append(x_dat[i])
                y_range.append(y_dat[i])
                e_range.append(e_dat[i])
    # np.array converts a list in a numpy array
    if (len(x_range)<10 or len(y_range)<0):
        print ('WARNING: Less than 10 points for the fitting in the function fit_range.')
    if (len(x_range)==0 or len(y_range)==0):
        print ('ERROR: No points for the fitting in the function fit_range.')

    x = np.array(x_range)
    y = np.array(y_range)
    e = np.array(e_range)
    x = x.reshape(x.shape[0],1)
    y = y.reshape(y.shape[0],1)
    e = e.reshape(e.shape[0],1)
    limited = np.concatenate((x,y),axis=1)
    limited = np.concatenate((limited,e),axis=1)
    return limited
# End of fit_range
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def sigmoidal(Q,lowQ=0.4,highQ=0.2,Q0=7.0,dQ=2.4):
    """
    Calculate a sigmoidal function

        sigmoidal(q) = lowQ * delta + highQ * (1-delta)

    where delta = exp[(q-q0)/dq]

    Note that sigmoidal(infty) = highQ
              sigmoidal(-infty) = lowQ

    Parameter
    ---------
    - Q : Array of floats
        A list with the abcsissas, usually Q-scale
    - lowQ : float
        The limiting value of the function at Q-->-infty
    - highQ : float
        The limiting value of the function at Q-->infty
    - Q0 : float
        Position of the inflexion point of the sigmoidal function
    - dQ : float
        Width of the transition of the sigmoidal function

    Returns
    -------
    - array: list of floats
        The list contains the values of the sigmoildal function corresponding
        to the given abcissas

    Date: Sat Dec 23 2023
    Author: Gabriel Cuello
--------------------------------------------------------------------------------
    """
    delta = np.exp((np.array(Q)-Q0)/dQ)
    return delta*lowQ+(1.0-delta)*highQ
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------
def siglin(Q,lowQ=0.4,highQ=0.2,Q0=7.0,dQ=2.4,a1=0.001):
    """
    Calculate a sigmoidal function multiplied by a linear behaviour

        siglin(q) = (lowQ * delta + highQ * (1-delta))*(1+a1*q)

    where delta = exp[(q-q0)/dq]

    Note that sigmoidal(infty) = highQ*(1+a1*q)
              sigmoidal(-infty) = lowQ*(1+a1*q)

    Parameter
    ---------
    - Q : Array of floats
        A list with the abcsissas, usually Q-scale
    - lowQ : float
        The limiting value of the sigmoidal function at Q-->-infty
    - highQ : float
        The limiting value of the sigmoidal function at Q-->infty
    - Q0 : float
        Position of the inflexion point of the sigmoidal function
    - dQ : float
        Width of the transition of the sigmoidal function
    - a1 : float
        Linear coefficient

    Returns
    -------
    - array: list of floats
        The list contains the values of the siglin function corresponding
        to the given abcissas.

    Date: Sat Dec 23 2023
    Author: Gabriel Cuello
--------------------------------------------------------------------------------
    """
    delta = np.exp((np.array(Q)-Q0)/dQ)
    return delta*lowQ+(1.0-delta)*highQ
#--------1---------2---------3---------4---------5---------6---------7---------8


#--------1---------2---------3---------4---------5---------6---------7---------8
def inelastic(x,A=51.0,lowQ=0.4,Q0=7.0,dQ=2.4):
    """
    Calculate the inelastic behaviour

        inelastic(x) = lowQ * [(1 + A**2/(1+A**2) * delta ] / (1+delta)

    where delta = exp[(x-q0)/dq]

    Note that inelastic(infty) = lowQ * A**2 / (1+A)**2
        i.e. sigma_free = sigma_bound * A**2 / (1+A)**2
        or   b_free = b_bound * A / (1+A)

    Parameter
    ---------
    - x : Array of floats
        A list with the abcsissas, usually Q-scale
    - A : float
        The mass number (mass/neutron mass)
    - low-Q : float
        The limiting value of the function at Q-->0, i.e., sigma_bound/4/pi
    - Q0 : float
        Position of the inflexion point of the sigmoidal function
    - dQ : float
        Width of the transition of the sigmoidal function

    Returns
    -------
    - array: list of floats
        The list contains the values of the inelastic function corresponding to
        the given abcissas

    Date: Wed Dec 30 2020
    Modified: Sat Oct 09 2021
    Author: Gabriel Cuello
--------------------------------------------------------------------------------
    """
    delta = np.exp((np.array(x)-Q0)/dQ)
    return lowQ*(1+A**2/(1+A)**2*delta)/(1+delta)
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def Lorch(q,qmax=23.5):
    """
    Evaluate the Lorch function at a given value of Q.

    The Lorch function is defined as follows:
        If Q<0 or Q>=Qmax,     Lorch(Q)=0.0
        If 0<Q<Qmax,           Lorch(Q) = sin(pi*Q/Qmax) / (pi*Q/Qmax)
        If Q=0,                Lorch(0) = 1.0

    Parameters
    ----------
    - q: float (It can be a numpy array)
        Abcissa
    - qmax: float
        The function is 0 at Q = Qmax

    Returns
    -------
    - float: the Lorch function evaluated at Q
        If the input is a numpy array, the output it is too.

    Date: Wed Dec 30 2020
    Modified: Sat Oct 09 2021
    Author: Gabriel Cuello
---------------------------------------
    """
    if qmax<=0: # qmax must be positive, otherwise the function
                # produces an error message and returns -999
        print ('ERROR: Non positive value for qmax in Lorch function.')
        return -999

    if ((q<0.0) or (q>=qmax)): # Outside the range [0,qmax) returns 0
        return 0.0

    elif q == 0.0: # Special case sin(x)/x = 1 for x = 0
        return 1.0
    else:
        a = q*np.pi/qmax # sin(x)/x
    return np.sin(a)/a
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def getSineIntegral(x):
    """
    This function evaluates the Sine Integral function:
        Si(x) = int_0^x sin(t)/t dt

    Parameters
    ----------
    - x: float
        Abcissa

    Returns
    -------
    - float: the Sine Integral function at x

    Requires:
    - integrate module from scipy

    Date: Dec 15, 2022
    Autor: Gabriel Cuello
    """
    npoints = 10000
    si = [] # A list containing the values of the window function
    t = []
    t.append(0.0)
    si.append(1.0) # A list containing the values of the window function
    for i in range(1,npoints):
        t.append(i*x/float(npoints))
        si.append(np.sin(t[i])/t[i])
    result = integrate.simps(si,t)
    return result
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def step(q,qmax=23.5):
    """
    Evaluate the Step function, which is basically a rectangle

    This is a rectangular window function
        returns 1 for 0 <= Q < Qmax
        returns 0 otherwise

    Parameters
    ----------
    - q: float (It can be a numpy array)
        Abcissa
    - qmax: float
        The rectangular function begins at Q=0 and stop at Qmax

    Returns
    -------
    - float: the Step function evaluated at Q
        If the input is a numpy array, the output it is too.

    Date Sat Oct 09 2021
    Author: Gabriel Cuello
---------------------------------------
    """
    if qmax<=0: # qmax must be positive, otherwise the function
                # produces an error message and returns -999
        print ('ERROR: Non positive value for qmax in step function.')
        return -999

    if ((q < 0.0) or (q >= qmax)): # Outside the range [0,qmax) returns 0
        return 0.0
    else:
        return 1.0
#--------1---------2---------3---------4---------5---------6---------7---------


#--------1---------2---------3---------4---------5---------6---------7-----
def LorchN(Q,Qmax=0):
    '''
    Evaluate the nornalised Lorch function at a given value of Q.

    The Lorch function is defined as follows:
        If Q<0 or Q>=Qmax,     Lorch(Q)=0.0
        If 0<Q<Qmax,           Lorch(Q) = sin(pi*Q/Qmax) / (pi*Q/Qmax)
        If Q=0,                Lorch(0) = 1.0

    Date: Wed Dec 30 2020
    Modified: Sat Oct 09 2021
    Author: Gabriel Cuello
---------------------------------------
    '''
#   The default value of Qmax is 0
#   If Qmax less or equal to 0 or greater than the last element
#   then Qmax is the last element of the input Q array.
    if (Qmax <= 0) or (Qmax > Q[-1]):
        Qmax = Q[-1]

    lorch = np.ones(len(Q))
    for i in range(len(Q)):
#       Outside the range [0,Qmax), lorch=0
        if ((Q[i]<0.0) or (Q[i]>=Qmax)):
            lorch[i] = 0.0
        elif Q[i] != 0.0:
            a = Q[i]*np.pi/Qmax # sin(x)/x
            lorch[i] = np.sin(a)/a
        else:
            lorch[i] = 1.0

    # Integral of the Lorch function for normalisation
    # This integral could be evaluated using the function getSineIntegral
    integralLorch = integrate.simps(lorch,Q)

    for i in range(len(Q)):
        lorch[i] *= Qmax / integralLorch
    return lorch
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def Lorentzian(x,A,x0,gamma,bckg):
    """
    Evaluate a Lorentzian function.

    Lor(x) = bckg + A * gamma**2/4/((x-q0)**2+gamma**2/4)

    Parameters
    ----------
    - x: array of floats
        The range of abcissas where the function is evaluated
    - A: float
        The scale of the Lorentzian
    - x0: float
        The centre of the Lorentizan
    - gamma: float
        The width of the Lorentzian
    - bckg: float
        An additive constant

    Returns
    -------
    - list: A list (or numpy array) with the evaluated Lorentian function

    Date: Wed Dec 30 2020
    Author: Gabriel Cuello
---------------------------------------
    """
    q = np.array(x)
    lor = A * gamma*gamma /4.0 /((q-x0)*(q-x0)+gamma*gamma/4.0) + bckg
    return lor
#--------1---------2---------3---------4---------5---------6---------7---------


#--------1---------2---------3---------4---------5---------6---------7---------
def Lorentzian_error(x,A=1,x0=0,gamma=1,bckg=0,e_A=0,e_x0=0,e_gamma=0,e_bckg=0):
    """
    Evaluate a Lorentzian function.

    Lor(x) = bckg + A * gamma**2/4/((x-q0)**2+gamma**2/4)

    Parameters
    ----------
    - x: array of floats
        The range of abcissas where the function is evaluated
    - A: float
        The scale of the Lorentzian
    - x0: float
        The centre of the Lorentizan
    - gamma: float
        The width of the Lorentzian
    - bckg: float
        An additive constant

    Returns
    -------
    - list: A list (or numpy array) with the evaluated Lorentian function

    Date: Wed Dec 30 2020
    Author: Gabriel Cuello
---------------------------------------
    """
    q = np.array(x)
    gam2 = gamma/2.0
    lor = A * gam2**2 /((q-x0)**2+gam2**2) + bckg
    peak = (lor-bckg)/A
    error  = e_bckg**2 + peak**2 * e_A**2
    error += (A/gam2*peak)**2 * (1-peak)**2 * e_gamma**2
    error += (2*A*(q-x0)/gam2**2)**2 * peak**2 * e_x0*22
    error = np.sqrt(error)
    return lor,error
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def Gaussian(x,A,x0,sigma,bckg):
    """
    Evaluate a normalised Gaussian function.

    Gau(x) = bckg + A /sqrt(2pi)/sigma * exp(-(x-x0)^2/2/sigma^2)

    Parameters
    ----------
    - x: array of floats
        The range of abcissas where the function is evaluated
    - A: float
        The area of the Gaussian
    - x0: float
        The centre of the Gaussian
    - gamma: float
        The width of the Gaussian
    - bckg: float
        An additive constant

    Returns
    -------
    - list: A list (or numpy array) with the evaluated Gaussian function

    Date: Sat Dec 15 2022
    Author: Gabriel Cuello
---------------------------------------
    """
    q = np.array(x)
    gau = A/np.sqrt(2*np.pi)/sigma * np.exp(-(q-x0)**2/2.0/sigma**2) + bckg
    return gau
#--------1---------2---------3---------4---------5---------6---------7---------



#--------1---------2---------3---------4---------5---------6---------7---------
def Gaussian_error(x,A=1,x0=0,sigma=1,bckg=0,e_A=0,e_x0=0,e_sigma=0,e_bckg=0):
    """
    Evaluate a normalised Gaussian function.

    Gau(x) = bckg + A /sqrt(2pi)/sigma * exp(-(x-x0)^2/2/sigma^2)

    Parameters
    ----------
    - x: array of floats
        The range of abcissas where the function is evaluated
    - A: float
        The area of the Gaussian
    - x0: float
        The centre of the Gaussian
    - gamma: float
        The width of the Gaussian
    - bckg: float
        An additive constant

    Returns
    -------
    - list: A list (or numpy array) with the evaluated Gaussian function

    Date: Sat Dec 15 2022
    Author: Gabriel Cuello
---------------------------------------
    """
    q = np.array(x)
    gau = A/np.sqrt(2*np.pi)/sigma * np.exp(-(q-x0)**2/2.0/sigma**2) + bckg
    peak = (gau-bckg)/A
    error  = e_bckg**2 + peak**2 * e_A**2
    error += peak**2 * (A/sigma)**2 * ((q-x0)**2/sigma**2 - 1)**2 * e_sigma**2
    error += peak**2 * (A/sigma**2*(q-x0))**2 * e_x0**2
    error = np.sqrt(error)
    return gau,error
#--------1---------2---------3---------4---------5---------6---------7---------


#--------1---------2---------3---------4---------5---------6---------7---------
def LorGau(x,f0=1.0,eta=0.5,sigma=2.0,gamma=2.0,bckg=0.0):
    """
    Evaluate a bell-shaped function centred at 0.

    This function is obtained as a linear combination of Gaussian and
    Lorentzian functions.

    LorGau(x) = background + factor * ( eta * Gau(x) + (1-eta) * Lor(x) )

        Gau(x) = exp(-x**2/2/sigma**2)
        Lor(x) = gamma**2/4/(x**2+gamma**2/4)

    Note that LorGau(0) = background + factor

    Parameters
    ----------
    - x: list of floats
        Range of abcissas
    - f0: float
        Multiplicative factor
    - eta: float
        Weight of the Gaussian contribution in the range [0,1]
    - sigma: float
        Width of the Gaussian
    - gamma: float
        Width of the Lorentzian
    - bckg: float
        Additive constant

    Returns
    -------
    - list: list (or numpy array)
        List with the evaluated peak-shaped function

    Date: Wed Dec 30 2020
    Author: Gabriel Cuello
---------------------------------------
    """
    q = np.array(x)
#    lor = Lorentzian_error(x,A=1.0,
    lor = gamma*gamma /4.0 /(q*q+gamma*gamma/4.0)
    gau = np.exp(-1.0*q*q/2.0/sigma/sigma)
    return f0 * (eta * gau + (1.0-eta) * lor) + bckg
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def LorGau_error(x,f0=(1.0,0.0),eta=(0.5,0.0),
                 sigma=(2.0,0.0),gamma=(2.0,0.0),bckg=(0.0,0.0)):
    """
    Evaluate the error a bell-shaped function centred at 0.

    This function is and obtained as a linear combination of Gaussian and
    Lorentzian functions.

    LorGau(x) = background + factor * ( eta * Gau(x) + (1-eta) * Lor(x) )

        Gau(x) = exp(-x**2/2/sigma**2)
        Lor(x) = gamma**2/4/(x**2+gamma**2/4)

    Note that LorGau(0) = background + factor

    Parameters
    ----------
    - x: list of floats
        Range of abcissas
    - f0: tuple of 2 floats
        Multiplicative factor and error
    - eta: tuple of 2 floats
        Weight of the Gaussian contribution in the range [0,1] amd error
    - sigma: tuple of 2 floats
        Width of the Gaussian and error
    - gamma: tuple of 2 floats
        Width of the Lorentzian and error
    - bckg: tuple of 2 floats
        Additive constant and error

    Returns
    -------
    - list: list (or numpy array)
        List with the errors of the peak-shaped function

    Date: Sun Dec 31 2023
    Author: Gabriel Cuello
---------------------------------------
    """
#    q = np.array(x)
    lor,lorerr = Lorentzian_error(x,A=1,x0=0,gamma=gamma[0],bckg=0,e_A=0,e_x0=0,e_gamma=gamma[1],e_bckg=0)
    gau,gauerr = Gaussian_error  (x,A=1,x0=0,sigma=sigma[0],bckg=0,e_A=0,e_x0=0,e_sigma=sigma[1],e_bckg=0)
    lorgau = bckg[0] + f0[0] * (eta[0] * gau + (1.0-eta[0]) * lor)
    error = bckg[1]**2 + (gau-lor)**2 * eta[1]**2
    error += ((lorgau-bckg[0])/f0[0])**2 * f0[1]**2
    error += eta[0]**2 * gauerr**2 + (1-eta[0])**2 * lorerr**2
    error = np.sqrt(error)
    return lorgau,error
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def GaussianA(x,A,x0,sigma,asym=0):
    """
    Evaluate an asymmetric Gaussian function. Exponential asymmetry.

    GauA(x) = A /sqrt(2pi)/sigma * exp(-((x-x0)**2)/2.0/sigma**2)
              * exp(-asym*(x-x0))

    Parameters
    ----------
    - x: array of floats
        The range of abcissas where the function is evaluated
    - A: float
        The area of the Gaussian
    - x0: float
        The centre of the Gaussian
    - gamma: float
        The width of the Gaussian
    - asym: float
        Exponential rate to take into account some asymmetry

    Returns
    -------
    - list: list of floats
        A list (or numpy array) with the evaluated asymmetric Gaussian function

    Date: Dec 27 2022
    Author: Gabriel Cuello
---------------------------------------
    """
    s = np.array(x)
    gau = A/np.sqrt(2*np.pi)/sigma * np.exp(-((s-x0)**2)/2.0/sigma**2)
    exp = np.exp(-asym*(s-x0))
    return gau*exp
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def niPeaks10(x,I0,slope,quad,wavelength,twotheta0,
            A0,A1,A2,A3,A4,A5,A6,A7,A8,A9,
            G0,G1,G2,G3,G4,G5,G6,G7,G8,G9,
            S0,S1,S2,S3,S4,S5,S6,S7,S8,S9):
    """
    Evaluate a function with 10 asymmetric Gaussian peaks to modelise the
    first 10 reflections of the fcc structure.

    Parameters
    ----------
    - x: array of floats
        The range of abcissas where the function is evaluated
    - I0, slope, quad: 3 floats
        Parameters of the linear background
    - Ax, Gx, Sx: 3 floats for each one of the 10 Gaussian functions
        A, G and S are the area, asymmetry and sigma of each Gaussian peak.
        Note that the centers are not necessary, because they are calculated
        from the fcc structure
    - wavelength: float
        Incident wavelength in Å
    - 2theta0: float
        Zero-angle correction in degrees

    Returns
    -------
    - list: list of floats
        A list (or numpy array) with the evaluated model for the Ni

    Date: Jul 9 2023
    Author: Gabriel Cuello
    """
# The centroids of each peak are taken from the list of reflections of a fcc
# lattice. Note the value of the lattice constant (a=3.52024 Å) that sould not be
# changed unless having very good reasons to do so.
    C0,C1,C2,C3,C4,C5,C6,C7,C8,C9 = reflections_fcc(wavelength, twotheta0, lattice=3.52024)[:10]

# Here the modelised diffractogram is created adding to the background
# 10 asymmetric Gaussian functions
    diff = I0+slope*x+quad*x*x+(GaussianA(x,A0,C0,S0,G0)+GaussianA(x,A1,C1,S1,G1)+
                       GaussianA(x,A2,C2,S2,G2)+GaussianA(x,A3,C3,S3,G3)+
                       GaussianA(x,A4,C4,S4,G4)+GaussianA(x,A5,C5,S5,G5)+
                       GaussianA(x,A6,C6,S6,G6)+GaussianA(x,A7,C7,S7,G7)+
                       GaussianA(x,A8,C8,S8,G8)+GaussianA(x,A9,C9,S9,G9))
    return diff
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def self024(x, self0, q2, q4):
    # Curve fitting function
    return self0 + (q2 + q4 * x**2) * x**2
#--------1---------2---------3---------4---------5---------6---------7---------


#--------1---------2---------3---------4---------5---------6---------7---------
def polyQ2(x,a0,a1,a2):
    '''
    Evaluate a 2nd-degree polynomial.

    poly(x) = a0 + a1 * x + a2 * x**2

    Parameters
    ----------
    - x: array of floats
        The range of abcissas where the function is evaluated
    - a_i: 3 floats
        Polynomial coefficients

    Returns
    -------
    - list: A list (or numpy array) with the evaluated polynomial

    Date: Wed Dec 30 2020
    Author: Gabriel Cuello
---------------------------------------
    '''
    q = np.array(x)
    return (a2 * q + a1 ) * q + a0
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def polyQ4(x,a0,a1,a2,a3,a4):
    """
    '''
    Evaluate a 4th-degree polynomial.

    poly(x) = a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4

    Parameters
    ----------
    - x: array of floats
        The range of abcissas where the function is evaluated
    - a_i: 5 floats
        Polynomial coefficients

    Returns
    -------
    - list: A list (or numpy array) with the evaluated polynomial

    Date: Wed Dec 30 2020
    Author: Gabriel Cuello
---------------------------------------
    """
    q = np.array(x)
#    a0 + a1 * q + a2 * q*q + a3 * q*q*q + a4 * q*q*q*q
    return a0 + (a1 + (a2 + (a3 + a4 * q) * q) * q) * q
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def vanaQdep(x,a0,a1,a2,A=51.0,lowQ=0.4,Q0=7.4,dQ=2.4):
    """
vanaQdep function

This is a function that describes the general behaviour of the incoherent
Vanadium signal as function of the momentum transfer (Q).

    vanadium(Q) = polyQ2 (Q,a0,a1,a2) * inelastic(Q,A,lowQ,Q0,dQ)

The first factor takes into account the instrument related effects (resolution)
and the second one accounts for the Q dependence of the inelasticity.

Use:
    y = vanadium(x,A,lowQ,Q0,dQ,a0,a1,a2)

Input:
    - x: a range of x
    - a_i: polynomial coefficients
    - A,lowQ,Q0,dQ: parameters of the sigmoidal function (see help for the
      inelastic function)

Output:
    - A list with the values of the function

Requires:
    - polyQ2
    - inelastic

Created on Wed Dec 30 2020
@author: Gabriel Cuello
---------------------------------------
    """
    q = np.array(x)
    polynomial = polyQ2(q,a0,a1,a2)        # Polynomial contribution
    sigmoidal = inelastic(q,A=A,lowQ=lowQ,Q0=Q0,dQ=dQ) # Inelasticity effect
    return sigmoidal*polynomial
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def getDate():
    """
    Read the current date.

    Parameters
    ----------
    - None

    Returns
    -------
    - string: Contains the current date in the format:
        Dayname DD/MM/YYYY HH:MM:SS

    Author: Gabriel Cuello
    Date:   Jan 2022
---------------------------------------
    """
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    #    curr_date = datetime.today()
    #    day = ' '+calendar.day_name[curr_date.weekday()]+' '
    day = " " + calendar.day_name[now.weekday()] + " "
    current_datetime = day + now.strftime("%d/%m/%Y %H:%M:%S")
    return current_datetime
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def sf_fcc(h,k,l):
    """
    Calculate the structure factor for a fcc lattice for a given plane hkl.

    Parameters
    ----------
    - h, k, l: 3 integers
        The 3 indices of Miller corresponding to the plane to evaluate
    Returns
    -------
    - float: The structure factor

    Date: Jul 1, 2023
    Author: Gabriel Cuello
---------------------------------------
    """
    result = 1.0 + np.exp(-1j*np.pi*(k+l)) + np.exp(-1j*np.pi*(h+l)) + np.exp(-1j*np.pi*(h+k))
    return result
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def sf_bcc(h,k,l):
    """
    Calculate the structure factor for a bcc lattice for a given plane hkl.

    Parameters
    ----------
    - h, k, l: 3 integers
        The 3 indices of Miller corresponding to the plane to evaluate
    Returns
    -------
    - float: The structure factor

    Date: Jul 1, 2023
    Author: Gabriel Cuello
---------------------------------------
    """
    result = 1.0 + np.exp(-1j*np.pi*(h+k+l))
    return result
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def sf_sc(h,k,l):
    """
    Calculate the structure factor for a sc lattice for a given plane hkl.

    This function is not very interesting, always returns 1.

    Parameters
    ----------
    - h, k, l: 3 integers
        The 3 indices of Miller corresponding to the plane to evaluate
    Returns
    -------
    - float: The structure factor

    Date: Jul 1, 2023
    Author: Gabriel Cuello
---------------------------------------
    """
    return 1.0
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def reflections_fcc(wavelength=0.5, twotheta0=0.0, lattice=3.52024):
    """
    Produce a list of the angular positions for a fcc lattice.

    This function will be used to fit the diffractogram of the Nickel powder
    sample, varying the wavelength and the zero angle correction.

    Note that nickel has a fcc structure with lattice parameter a = 3.52024 Å.

    Parameters
    ----------
    - wavelength: float
        Wavelength, in Å. Default value = 0.5 Å.
    - twotheta0: float
        Zero angle correction, in degrees. Default value = 0 degrees
    - lattice: float
        Lattice parameter, in Å. Default value = 3.52024 Å.

    Returns
    -------
    - list: list of floats with the angular positions (in degrees) of all
        allowed reflections for a fcc structure, in the range of 0 to 180 deg.

    Date: Jul 1, 2023
    Author: Gabriel Cuello
---------------------------------------
    """
    moduleMax = int(2.0*lattice/wavelength)
    sinus = []
    for h in range(moduleMax):
        for k in range(moduleMax):
            for l in range(moduleMax):
                hkl = [h,k,l]
                module = np.sqrt(hkl[0]**2+hkl[1]**2+hkl[2]**2)
#                plane = "({} {} {})".format(*hkl)
                sf = sf_fcc(*hkl).real
                if sf*module > 0:
                    sintheta = module*wavelength/2.0/lattice
                    if sintheta < 1.0:
                        sinus.append(sintheta)
    rad_fcc = 2.0 * np.arcsin(sorted(set(sinus)))
    deg_fcc1 = twotheta0 + np.array(180.0/np.pi * rad_fcc)
    deg_fcc2 = -twotheta0 + np.array(sorted(180.0 - deg_fcc1))
    deg_fcc = np.concatenate([deg_fcc1, deg_fcc2])
    return deg_fcc
#--------1---------2---------3---------4---------5---------6---------7---------


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#
#  Fourier Transforms
#
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

#--------1---------2---------3---------4---------5---------6---------7-----
def getSineFT(vvec,yvec,uvec,vmax=0,c=1.0,s=1.0,w=0):
    ''''
    Calculate a Sine Fourier Transform as follows:
            c * int_0^vmax (y-s) v sin(v*u) * win(v) dv

    Parameters
    ----------
    - vvec: array of floats
        Variable over which the integral is performed.
        In a direct transformation this variable is Q.
        In a reverse transformation this variable is R.
    - yvec: array of floats
        The structure factor S(Q) in the direct transformation or
        the pair distribution function g(R) in a reverse transformation.
    - uvec: array of floats
        Conjugated variable of v. This is R in adirect transformation or
        Q in the reverse tranformation.
    - vmax: float
        The maximum value of the integration variable. If zero or lower, or
        bigger than the last value in the range, it is the biggest value in
        the range. Default value = 0.
    - c: float
        Multiplicative constant. Default value = 1.
    - s: float
        Additive constant to be subracted from the S(Q) or g(R). In general,
        this constant should be 1. Default value = 1.
    - w: integer
        This allows to chooes the windo function to apply to the calculation.
        w = 0 : No window function (default value)
        w = 1 : Normalised Lorch function

    Returns
    -------
    -array: Array containing the Fourier Transformed function. It has the same
            dimension of the conjugated variable u.

    Date: Jul 1 2023
    Author: Gabriel Cuello
---------------------------------------
    '''
# Number of points in the function to be transformed
    nbr_v = len(vvec)  # abcissa
    nbr_y = len(yvec)  # ordinate
    # These numbers must be equal, otherwise print error message
    if nbr_v != nbr_y:
        print ('ERROR: abcissa and ordinate do not have the same dimension')
        print ('       in the Fourier Transform function (getSineFT).')
        return None

#   The default value of vmax is 0
#   If vmax less or equal to 0 or greater than the last element
#   then vmax is the last element of the input v array.
    if (vmax <= 0) or (vmax > vvec[-1]):
        vmax = vvec[-1]

    win=np.ones(nbr_v)
    if w == 1:
        win = LorchN(vvec,vmax)

    stf = np.zeros(len(uvec))
    for i in range(len(uvec)):
#        integ = soq*q*np.sin(q*r)
        integ = (yvec-s)*vvec*np.sin(vvec*uvec[i])*win
#        result = integrate.simps(integ,q)
#        pcf.append(result)
#        integ = soq*q*np.sin(q*r)*win
        stf[i] = 2.0 / np.pi * c * integrate.simps(integ,vvec)
    # Pair correlation function or G(r)
    return stf
#--------1---------2---------3---------4---------5---------6---------7-----

#--------1---------2---------3---------4---------5---------6---------7---------
def sineFT(StrFactor,nr,qmax=23.5,density=0.1,constant=1.0,selfsca=1.0,window=0):
    """
    Perform the sinus Fourier transform of the structure factor.

    The integral is performed from 0 to Qmax, and the integrand is:
        constant * (S(Q)-self) * Q * sin(Q*R) * window(Q)

    Parameter
    ---------
    - StrFactor: matrix of floats (2 columns)
        First column: Q-scale in 1/Å
        Second column: Structure factor
    nr: array of floats
        The R-scale, in Å
    - qmax: float
        Upper limit of the integral (in 1/Å)
    - density: float
        Atomic density of the sample (in atoms/Å3)
    - constant: float
        Multiplicative constant (it should be 1)
    - selfsca: float
        Additve constant (it should be 1)
    - window: 0 or 1, for step or Lorch window function, respectively

    Returns
    -------
    - matrix: floats in 6 columns
        col1: R
        col2: pcf, pair correlation function
        col3: pdf, pair distribution function
        col4: rdf, radial distribution function or linear density
        col5: tor, rdf/R, which produces more symmetrical peaks
        col6: run, running integral of the rdf, i.e., integral from 0 to R

    Date: Wed Dec 30 2020
    Modified: Sat Oct 09 2021
    Author: Gabriel Cuello
---------------------------------------
    """
    soq = StrFactor[:,1]
    q = StrFactor[:,0]
    pcf = []
    pdf = []
    rdf = []
    tor = []
    run = []
#    infile = np.genfromtxt(filename, skip_header= 7) #creates an array from xA, yA, zA
#    q  = infile[:, 0]
#    soq  = infile[:, 1]
#    #err  = infile[:, 2]

    win = [] # A list containing the values of the window function

    if window ==0:
        for qu in q:
            win.append(step(qu,qmax))
    else:
        for qu in q:
            win.append(Lorch(qu,qmax))
#       Integral of the Lorch function for normalisation
        integralLorch = integrate.simps(win,q)
#        print ('Normalisation of Lorch function: ',integralLorch/qmax)
        for i in range(len(win)):
            win[i] = win[i] * qmax / integralLorch

    deltaR = nr[1]-nr[0]
    for r in nr:
#        integ = soq*q*np.sin(q*r)
        integ = (soq-selfsca)*q*np.sin(q*r)*win
#        result = integrate.simps(integ,q)
#        pcf.append(result)
#        integ = soq*q*np.sin(q*r)*win
        result = 2.0 / np.pi * constant * integrate.simps(integ,q)
    # Pair correlation function or G(r)
        pcf.append(result)
    # Pair distribution function or g(r)
        if r <= 0:
            pdf.append(0.0)
        else:
            pdf.append(result / 4.0 / np.pi / r /density + 1.0)
    # Radial distribution function or RDF(r)
        rdf.append(result * r + 4.0 * np.pi * density * r**2)
    # T(r) = RDF(r)/r; symmetric peaks for fitting
        tor.append(result + 4.0 * np.pi * density * r)
    # Running integral of the RDF(r), i.e., integral from 0 to r
    for r in nr:
        ind = 1+int(r/deltaR)
        xr = nr[0:ind]
        integ = np.array(rdf)[0:ind]
        result = integrate.simps(integ,xr)
        run.append(result)
    rrr = np.array(nr)
    pcf = np.array(pcf)
    pdf = np.array(pdf)
    rdf = np.array(rdf)
    tor = np.array(tor)
    run = np.array(run)
    rrr = rrr.reshape(rrr.shape[0],1)
    pcf = pcf.reshape(pcf.shape[0],1)
    pdf = pdf.reshape(pdf.shape[0],1)
    rdf = rdf.reshape(rdf.shape[0],1)
    tor = tor.reshape(tor.shape[0],1)
    run = run.reshape(run.shape[0],1)
    fou = np.concatenate((rrr,pcf),axis=1)
    fou = np.concatenate((fou,pdf),axis=1)
    fou = np.concatenate((fou,rdf),axis=1)
    fou = np.concatenate((fou,tor),axis=1)
    fou = np.concatenate((fou,run),axis=1)
    return fou
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def backFT(qu,ar,pdf,density,cut):
    ''''
    Calculate the back Fourier Transform of the Pair Distribution Function.

    Parameters
    ----------
    - qu: array of floats
        Q-scale
    - ar: array of floats
        R-scale
    - pdf: array of floats
        Partial Distribution Function (g(R))
    - density: float
        Atomic density (atoms/A3)
    - cut: list of floats
        This list has 1 or 3 elements. The g(R) is forced to be 0
        1. Between 0 and cut[0]
        2. Between cut[1] and cut[2]

    Returns
    -------
    -lists: Six list of floats
        soq: the new structure factor
        pcf: the new pair correlation function
        pdf: the new pair distribution function
        rdf: the new radial distribution function
        tor: the new total distribution function
        run: The new running integral of the rdf

    Date: Jul 1 2023
    Author: Gabriel Cuello
---------------------------------------
    '''
    pdf_cut  = pdf.copy()
    if len(cut) == 1 and cut[0] == -1:
        for i in range(len(pdf)-1,0,-1):
            if pdf[i] < 0:
                break
        for j in range(i+1):
            pdf_cut[j] = 0.0
    elif len(cut) == 1 and cut[0] > 0.0:
        for i in range(len(pdf)):
            if ar[i] < cut[0]:
                pdf_cut[i] = 0.0
    elif len(cut) == 3:
        for i in range(len(pdf)):
            if ar[i] < cut[0] or (cut[1] < ar[i] and ar[i] < cut[2]):
                pdf_cut[i] = 0.0
#--------1---------2---------3---------4---------5---------6---------7-----
# Calculation of the structure factor as the back Fourier transform of the
# given pdf
    soq = 1+2.0*np.pi**2*density/qu*getSineFT(ar,pdf_cut,qu,w=0)
#--------1---------2---------3---------4---------5---------6---------7-----
# Pair correlation function, G(R)
    pcf  = getSineFT(qu,soq,ar,w=0)
#--------1---------2---------3---------4---------5---------6---------7-----
# Pair distribution function, g(R)
    pdf  = 1+pcf /4.0/np.pi/density/ar
# Radial distribution function, RDF(R)
    rdf  = 4.0*np.pi*density*ar*ar*pdf
# Linearised radial distribution function, T(R) = RDF(R)/R
    tor  = rdf /ar
# Running integral of the radial distribution function
    run  = np.zeros(len(ar))
    for i in range(1,len(ar)):
        run[i]  = integrate.simps(rdf[0:i], ar[0:i])
    return soq,pcf,pdf,rdf,tor,run
#--------1---------2---------3---------4---------5---------6---------7-----


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#
#  Input Output
#
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

#--------1---------2---------3---------4---------5---------6---------7---------8
def plotQ(plt,title=None,xmin=0,xmax=25,ymin=None,ymax=None,
          ylabel='Intensity (arb. units)'):
    plt.title(title)
    plt.xlabel(r'$Q ({\rm \AA}^{-1}$)')
    plt.ylabel(ylabel)
    plt.axis([xmin,xmax,ymin,ymax])
    plt.grid(True)
    plt.legend(loc='best')
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def plotA(plt,title=None,xmin=0,xmax=140,ymin=None,ymax=None,
         ylabel='Intensity (arb. units)'):
    plt.title(title)
    plt.xlabel(r'$2\theta$ ($^o$)')
    plt.ylabel(ylabel)
    plt.axis([xmin,xmax,ymin,ymax])
    plt.grid(True)
    plt.legend(loc='best')
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def plotR(plt,title=None,xmin=0,xmax=20,ymin=None,ymax=None,
         ylabel='Intensity (arb. units)'):
    plt.title(title)
    plt.xlabel(r'$R$ (Å)')
    plt.ylabel(ylabel)
    plt.axis([xmin,xmax,ymin,ymax])
    plt.grid(True)
    plt.legend(loc='best')
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def getRunningParams(parfile):
    """
    Read the parameters file for data treatment notebooks.

    Parameters
    ----------
    - parfile: a string containing the name of the parameters file.

    Returns
    -------
  - inputPar: dictionary:
        Contains all the parameters read from the parameters file

    Description of the keywords in the parameters file.
        ------------------------------------------------------------------
        Keyword     Description                             Parameter file
        ------------------------------------------------------------------
        <par>       Parameter file                          H2O.par
        <ins>       Instrument                              D4
        <pro>       Proposal number                         5-23-796
        <mpr>       Main proposer                           Main proposer name
        <lco>       Local contact                           LC name
        <dte>       Dates start-end                         29/10/2023-31/10/2023
        <sta>       Start date
        <end>       End date
        <cyc>       Reactor cycle                           234
        <log>       Notebook book-page                      42-210
        <nbk>       Notebook number
        <pag>       Page number
        <env>       Sample environment                      CF
        <zac>       Zero angle correction (deg)             -0.166420
        <wle>       Wavelength (Å)                          0.4988687
        <sli>       Slits (mm). Left(lohen) Right(gams)     6.0 -6.0
        <lft>       Left slit (downstream direction)
        <rgt>       Right slit (downstream direction)
        <fla>       Flags (mm). Top Bottom                  25.0 -25.0
        <top>       Top flag
        <bot>       Bottom flag
        <apr>       Angular precision (deg)                 0.125
        <asc>       Angular scale (deg). ini fin step       0.0 140.0 0.125
        <qsc>       Q scale (1/Å). ini fin step             0.0 23.5 0.02
        <rsc>       r scale (Å). ini fin step               0.0 50.0 0.01
        <Cmater>    Mat. container: SiO2,V,TiZr,Nb,None     V
        <Cshape>    Shape innerD outerD height (mm)         cylinder 4.8 5.0 60.0
        <Amater>    Material absorber: B, B4C, None         B
        <Ashape>    Shape innerD outerD height (mm)         cylinder 4.8 5.0 60.0
        <Nmater>    Material vanadium: V, None              V
        <Nshape>    Shape innerD outerD height (mm)         cylinder 0.0 6.08 60.0
        <Emater>    Material environment: V, Nb, W, A       V
        <Eshape>    Shape innerD outerD height (mm)         cylinder 24.8 25.0 100.0
        <Sdescr>    Sample description                      Li1.25(Fe0.33Ti0.3Nb0.1)(O2)
        <StempK>    Sample Temperature (in K)               300.0
        <Sato01>    Atom 1 Symbol number                    Li 125.0
        <Sato02>    Atom 2 Symbol number                    Ti  30.0
        <Sato03>    Atom 3 Symbol number                    F    0.0
        ...          ...                                    ..  ...
        <SatoNN>    Atom NN Symbol number                   Nb  10.0
        <SNelem>    Number of elements
        <Sdegcc>    Density (g/cm3)                         5.1
        <Smassc>    Mass in the container (g)               2.1271
        <Sshape>    Shape, diameter, height (in mm)         cylinder 0.0 4.8 59.5
        <Sfulln>    Sample fullness                         1.0
        <Stitle>    Title (for plots and output files)      Li_noF
        <CoFile>    Container file                          ../regdata/MTcan.qdat
        <AbFile>    Absorber file                           ../regdata/MTbelljar.qdat
        <NoFile>    Normaliser file                         ../regdata/vanadium.qdat
        <EnFile>    Environment file                        ../regdata/MTbelljar.qdat
        <SaFile>    Sample file                             ../regdata/Li_noF.qdat

    Some parameters are not given in the parameters file, because they are taken
    from others. For example, the number of chemical species (SNelem), which is
    determined by the number of different atoms specified in the file.

    Date: 22/02/2023
    Author: Gabriel Cuello
--------------------------------------------------------------------------------
    """
#--------1---------2---------3---------4---------5---------6---------7---------8
    shapes =       ['cylinder']
    containers =   ['SiO2','V','TiZr','Nb','None']
    absorbers =    ['B','None']
    normalisers =  ['V','None']
    environments = ['V','A','None']
    Nelements = 0  # Number of elements (or chemical species) in the sample

    inputPar = {} # Initialising the dictionary
    inputPar["<par>"] = ('Parameter file   ',parfile)
    with open(parfile, "r") as f:
        lines = f.readlines()

# Testing the first character of each non-blank line.
# If this character is not #, ! or <, stops the program.
    for i in range(len(lines)):
        if (len(lines[i]) > 1):
            first = (lines[i][0] != "#") and (lines[i][0] != "!") and (lines[i][0] != "<")
            if first:
                print ("Wrong input on line: ",i+1," in file: ",parfile)
                sys.exit()

# Loop over all lines in the input file
    for i in range(len(lines)):
    #    print (lines[i][0:-1])
        if (lines[i][0] == "#") or (lines[i][0] == "!"):
    #        print ("Comment line",len(lines[i]))
            pass
        elif len(lines[i]) == 1:
    #        print("Blank line")
            pass
        elif lines[i][0] == "<":
            line = lines[i].split(" ")
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<inst>"] = ('Instrument   ','D4')
            if line[0] == "<ins>":
                inputPar[line[0]] = ('Instrument:',line[1])
#                print ("{}: {}"
#                       .format(inputPar[line[0]][0],inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<pro>"]= ('Proposal     ','0-00-0000')
            if line[0] == "<pro>":
                inputPar[line[0]] = ('Proposal:',line[1])
#                print ("{}: {}"
#                       .format(inputPar[line[0]][0],inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<mpr>"]= ('Main Proposer','Nobody')
            if line[0] == "<mpr>":
                inputPar[line[0]] = ('Main Proposer:',line[1])
#                print ("{}: {}"
#                       .format(inputPar[line[0]][0],inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<lco>"]= ('Local Contact','Cuello')
            if line[0] == "<lco>":
                inputPar[line[0]] = ('Local Contact:',line[1])
#                print ("{}: {}"
#                       .format(inputPar[line[0]][0],inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<dte>"]= ('Dates        ','01/01/2000-31/12/2000')
#            inputPar["<sta>"]= ('Starting date','01/01/2000')
#            inputPar["<end>"]= ('Ending date  ','31/12/2000')
            if line[0] == "<dte>":
                inputPar[line[0]] = ('Dates:',line[1])
                hyphen = line[1].find('-')
                inputPar['<sta>'] = ('Starting date:',line[1][0:hyphen])
                inputPar['<end>'] = ('Ending date:',line[1][hyphen+1:])
#                print ("{}: {}"
#                       .format(inputPar['<sta>'][0],inputPar['<sta>'][1]))
#                print ("{}: {}"
#                       .format(inputPar['<end>'][0],inputPar['<end>'][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<cyc>"]= ('Cycle        ','230')
            if line[0] == "<cyc>":
                inputPar[line[0]] = ('Cycle:',line[1])
#                print ("{}: {}"
#                       .format(inputPar[line[0]][0],inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<log>"]= ('Logbook      ','00-00')
#            inputPar["<nbk>"]= ('Notebook nbr.','00')
#            inputPar["<pag>"]= ('Page         ','00')
            if line[0] == "<log>":
                inputPar[line[0]] = ('Logbook:',line[1])
                hyphen = line[1].find('-')
                inputPar['<nbk>'] = ('Notebook #:',line[1][0:hyphen])
                inputPar['<pag>'] = ('Page:',line[1][hyphen+1:])
#                print ("{}: {}"
#                       .format(inputPar['<nbk>'][0],inputPar['<nbk>'][1]))
#                print ("{}: {}"
#                       .format(inputPar['<pag>'][0],inputPar['<pag>'][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<env>"]= ('Environment  ','A')
            if line[0] == "<env>":
                inputPar[line[0]] = ('Environment:',line[1])
#                print ("{}: {}"
#                       .format(inputPar[line[0]][0],inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<zac>"]= ('Zero angle   ',0.0)
            if line[0] == "<zac>":
                inputPar[line[0]] = ('Zero angle (deg):',float(line[1]))
#                print ("{}: {} deg"
#                       .format(inputPar[line[0]][0],inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<wle>"]= ('Wavelength   ',0.5)
            if line[0] == "<wle>":
                inputPar[line[0]] = ('Wavelength (Å):',float(line[1]))
#                print ("{}: {} Å"
#                       .format(inputPar[line[0]][0],inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<sli>"]= ('Slits        ','6.0 -6.0')
#            inputPar["<lft>"]= ('Left (Lohen) ',6.0)
#            inputPar["<rgt>"]= ('Right (Gams) ',-6.0)
            if line[0] == "<sli>":
                inputPar[line[0]] = ('Slits (mm):',line[1])
                inputPar['<lft>'] = ('Left (mm):',float(line[1]))
                inputPar['<rgt>'] = ('Right (mm):',float(line[2]))
#                print ("{}: {} mm"
#                       .format(inputPar['<lft>'][0],inputPar['<lft>'][1]))
#                print ("{}: {} mm"
#                       .format(inputPar['<rgt>'][0],inputPar['<rgt>'][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<fla>"]= ('Flags        ','25.0 -25.0')
#            inputPar["<top>"]= ('Top          ',25.0)
#            inputPar["<bot>"]= ('Bottom       ',-25.0)
            if line[0] == "<fla>":
                inputPar[line[0]] = ('Flags (mm):',line[1])
                inputPar['<top>'] = ('Top (mm):',float(line[1]))
                inputPar['<bot>'] = ('Bottom (mm):',float(line[2]))
#                print ("{}: {} mm"
#                       .format(inputPar['<top>'][0],inputPar['<top>'][1]))
#                print ("{}: {} mm"
#                       .format(inputPar['<bot>'][0],inputPar['<bot>'][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<apr>"]=      ('Angular prec.',0.125)
            if line[0] == "<apr>":
                inputPar[line[0]] = ('Angular precision (deg):',float(line[1]))
#                print ("{}: {} deg"
#                       .format(inputPar[line[0]][0],inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<asc>"]=      ('Angular scale',0,140.0,0.125)
            if line[0] == "<asc>":
                inputPar[line[0]] = ('Angular scale (deg):',float(line[1]),
                                     float(line[2]),float(line[3]))
#                print ("{}: from {} deg to {} deg in steps of {} deg"
#                       .format(inputPar[line[0]][0],inputPar[line[0]][1],
#                               inputPar[line[0]][2],inputPar[line[0]][3]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<qsc>"]=      ('Q       scale',0,23.4,0.02)
            if line[0] == "<qsc>":
                inputPar[line[0]] = ('Q scale (1/Å):',float(line[1]),
                                     float(line[2]),float(line[3]))
#                print ("{}: from {} 1/Å to {} 1/Å in steps of {} 1/Å"
#                       .format(inputPar[line[0]][0],inputPar[line[0]][1],
#                               inputPar[line[0]][2],inputPar[line[0]][3]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<asc>"]=      ('r       scale',0,20.0,0.01)
            if line[0] == "<rsc>":
                inputPar[line[0]] = ('r scale (Å):',float(line[1]),
                                     float(line[2]),float(line[3]))
#                print ("{}: from {} Å to {} Å in steps of {} Å"
#                       .format(inputPar[line[0]][0],inputPar[line[0]][1],
#                               inputPar[line[0]][2],inputPar[line[0]][3]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<Cmater>"]=   ('Container material  ','V')
            if line[0] == "<Cmater>":
#                print('Found <Cmater>')
                inputPar[line[0]] = ('Container material:',line[1])
                if line[1] in containers:
#                    print ("{}: {}"
#                           .format(inputPar[line[0]][0],inputPar[line[0]][1]))
                    pass
                else:
                    print ("The container {} is not available"
                           .format(inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<Cshape>"]=   ('Container shape     ','cylinder', 4.8, 5.0, 60.0)
            if line[0] == "<Cshape>":
                inputPar[line[0]] = ('Container shape:',line[1],line[2],
                                     line[3],line[4])
                if line[1] in shapes:
#                    print ("{}: A {} with {} mm (id) {} mm (od) and {} mm height"
#                           .format(inputPar[line[0]][0],inputPar[line[0]][1],
#                                   inputPar[line[0]][2],inputPar[line[0]][3],
#                                   inputPar[line[0]][4]))
                    pass
                else:
                    print ("The shape {} is not available"
                           .format(inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<Amater>"]=   ('Absorber material   ','B')
            if line[0] == "<Amater>":
                inputPar[line[0]] = ('Absorber material:',line[1])
                if line[1] in absorbers:
#                    print ("{}: {}"
#                           .format(inputPar[line[0]][0],inputPar[line[0]][1]))
                    pass
                else:
                    print ("The absorber {} is not available"
                           .format(inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<Ashape>"]=   ('Absorber shape      ','cylinder', 0.0, 4.8, 60.0)
            if line[0] == "<Ashape>":
                inputPar[line[0]] = ('Absorber shape:',line[1],line[2],
                                     line[3],line[4])
                if line[1] in shapes:
#                    print ("{}: A {} with {} mm (id) {} mm (od) and {} mm height"
#                           .format(inputPar[line[0]][0],inputPar[line[0]][1],
#                                   inputPar[line[0]][2],inputPar[line[0]][3],
#                                   inputPar[line[0]][4]))
                    pass
                else:
                    print ("The shape {} is not available"
                           .format(inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<Nmater>"]=   ('Normaliser material   ','V')
            if line[0] == "<Nmater>":
                inputPar[line[0]] = ('Normaliser material:',line[1])
                if line[1] in normalisers:
#                    print ("{}: {}"
#                           .format(inputPar[line[0]][0],inputPar[line[0]][1]))
                    pass
                else:
                    print ("The normaliser {} is not available"
                           .format(inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<Nshape>"]=   ('Vanadium shape      ','cylinder', 0.0, 6.08, 60.0)
            if line[0] == "<Nshape>":
                inputPar[line[0]] = ('Normaliser shape:',line[1],line[2],
                                     line[3],line[4])
                if line[1] in shapes:
#                    print ("{}: A {} with {} mm (id) {} mm (od) and {} mm height"
#                           .format(inputPar[line[0]][0],inputPar[line[0]][1],
#                                   inputPar[line[0]][2],inputPar[line[0]][3],
#                                   inputPar[line[0]][4]))
                    pass
                else:
                    print ("The shape {} is not available"
                           .format(inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<Emater>"]=   ('Environment material','V')
            if line[0] == "<Emater>":
                inputPar[line[0]] = ('Environment material:',line[1])
                if line[1] in environments:
#                    print ("{}: {}"
#                           .format(inputPar[line[0]][0],inputPar[line[0]][1]))
                    pass
                else:
                    print ("The environment {} is not available"
                           .format(inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<Eshape>"]=   ('Environment shape   ','cylinder', 24.9, 25.0, 250.0)
            if line[0] == "<Eshape>":
                inputPar[line[0]] = ('Environment shape:',line[1],line[2],
                                     line[3],line[4])
                if line[1] in shapes:
#                    print ("{}: A {} with {} mm (id) {} mm (od) and {} mm height"
#                           .format(inputPar[line[0]][0],inputPar[line[0]][1],
#                                   inputPar[line[0]][2],inputPar[line[0]][3],
#                                   inputPar[line[0]][4]))
                    pass
                else:
                    print ("The Environment {} is not available"
                           .format(inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<Sdescr>"]=  ('Sample description  ',"Your sample")
            if line[0] == "<Sdescr>":
                description = lines[i][9:]
                hash = description.find('#')
                inputPar[line[0]] = ('Sample description:  ',description[0:hash])
#                print ("{}: {}"
#                       .format(inputPar[line[0]][0],inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<StempK>"]=   ('Sample temperature  ',300.0)
            if line[0] == "<StempK>":
                inputPar[line[0]] = ('Sample temperature:  ',float(line[1]))
#                print ("{}: {} K"
#                       .format(inputPar[line[0]][0],inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<Sato01>"] = ('Atom 01',1.0)
#           inputPar["<SNelem>"] = ('Sample nbr. elements',1.0)
            if line[0][0:5] == "<Sato":
                Nelements += 1
                usefulinfo = lines[i][9:]
                hash = usefulinfo.find('#')
                # remove blank before and after the useful info, and split it
                atomdat = (usefulinfo[0:hash].strip()).split(" ")
                # new list without empty strings
                atompar = [x for x in atomdat if x != '']
                inputPar[line[0]] = ('Atom '+str(Nelements).zfill(2)+': ',*atompar)
                inputPar['<SNelem>'] = ('Number of elements:',Nelements)
#                print (*inputPar[line[0]])
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<Sdegcc>"] =  ('Sample density      ',1.0)
            if line[0] == "<Sdegcc>":
                inputPar[line[0]] = ('Sample density:      ',float(line[1]))
#                print ("{}: {} g/cm3"
#                       .format(inputPar[line[0]][0],inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<Smassc>"] =  ('Sample mass in can  ',2.0)
            if line[0] == "<Smassc>":
                inputPar[line[0]] = ('Sample mass in can:  ',float(line[1]))
#                print ("{}: {} g"
#                       .format(inputPar[line[0]][0],inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<Sshape>"] =  ('Sample shape        ','cylinder',0.0,4.8,60.0)
            if line[0] == "<Sshape>":
                inputPar[line[0]] = ('Sample shape:',line[1],line[2],
                                     line[3],line[4])
                if line[1] in shapes:
#                    print ("{}: A {} with {} mm (id) {} mm (od) and {} mm height"
#                            .format(inputPar[line[0]][0],inputPar[line[0]][1],
#                                   inputPar[line[0]][2],inputPar[line[0]][3],
#                                   inputPar[line[0]][4]))
                    pass
                else:
                    print ("The shape {} is not available"
                           .format(inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<Sfulln>"] =  ('Sample fullness     ',1.0)
            if line[0] == "<Sfulln>":
                inputPar[line[0]] = ('Sample fullness:',float(line[1]))
#                print ("{}: {} g"
#                       .format(inputPar[line[0]][0],inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<Stitle>"] =  ('Sample title        ','Your title')
            if line[0] == "<Stitle>":
                inputPar[line[0]] = ('Sample title:        ',line[1])
#                print ("{}: {}"
#                       .format(inputPar[line[0]][0],inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            if inputPar["<Cmater>"] != 'None':
            if line[0] == "<CoFile>":
                inputPar[line[0]] = ('Container file:      ',line[1])
#                print ("{}: {}"
#                       .format(inputPar[line[0]][0],inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            if inputPar["<Amater>"] != 'None':
            if line[0] == "<AbFile>":
                inputPar[line[0]] = ('Absorber file:       ',line[1])
#                print ("{}: {}"
#                       .format(inputPar[line[0]][0],inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            if inputPar["<Nmater>"] != 'None':
            if line[0] == "<NoFile>":
                inputPar[line[0]] = ('Vanadium file:       ',line[1])
#                print ("{}: {}"
#                       .format(inputPar[line[0]][0],inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            if inputPar["<Emater>"] != 'None':
            if line[0] == "<EnFile>":
                inputPar[line[0]] = ('Environment file:    ',line[1])
#                print ("{}: {}"
#                       .format(inputPar[line[0]][0],inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
#            inputPar["<SaFile>"] = ('Sample file         ',"dummySmpl.qdat")
            if line[0] == "<SaFile>":
                inputPar[line[0]] = ('Sample file:         ',line[1])
#                print ("{}: {}"
#                       .format(inputPar[line[0]][0],inputPar[line[0]][1]))
#--------1---------2---------3---------4---------5---------6---------7---------8
        else:
            print ("Input error in line: ",i+1," file: ",parfile)
            sys.exit()
    return inputPar
#--------1---------2---------3---------4---------5---------6---------7---------8

class Measurement:
    '''
An object of this class contains all input data for a given sample.

    Parameters
    ----------
    - inputPar: dictionary
        This dictionary contains all the paramters read from the parameters
        file. In fact, it is the dictionary created by the function
        getRunningParams().
        For a list of keys and their meaning: help(getRunningParams)

Date: 24/02/2023
Author: Gabriel Cuello
    '''
    def __init__(self,inputPar):
        self.inputPar = inputPar
        self.parfile = inputPar['<par>'][1]
        self.instrument = inputPar['<ins>'][1]
        self.proposal = inputPar['<pro>'][1]
        self.mainProposer = inputPar['<mpr>'][1]
        self.localContact = inputPar['<lco>'][1]
        self.startDate = inputPar['<sta>'][1]
        self.endDate = inputPar['<end>'][1]
        self.cycle = inputPar['<cyc>'][1]
        self.notebook = inputPar['<nbk>'][1]
        self.page = inputPar['<pag>'][1]
        self.envCode = inputPar['<env>'][1]
        self.zeroAngle = float(inputPar['<zac>'][1])
        self.wavelength = float(inputPar['<wle>'][1])
        self.vslits = (float(inputPar['<lft>'][1]),
                       float(inputPar['<rgt>'][1]))
        self.hslits = (float(inputPar['<top>'][1]),
                       float(inputPar['<bot>'][1]))
        self.beamHeight = self.hslits[0]-self.hslits[1]
        self.beamWidth  = self.vslits[0]-self.vslits[1]
        self.angularPrecision = float(inputPar['<apr>'][1])
        self.aScale = (float(inputPar['<asc>'][1]),
                       float(inputPar['<asc>'][2]),
                       float(inputPar['<asc>'][3]))
        self.qScale = (float(inputPar['<qsc>'][1]),
                       float(inputPar['<qsc>'][2]),
                       float(inputPar['<qsc>'][3]))
        self.rScale = (float(inputPar['<rsc>'][1]),
                       float(inputPar['<rsc>'][2]),
                       float(inputPar['<rsc>'][3]))
        self.container = (inputPar['<Cmater>'][1], inputPar['<Cshape>'][1],
                          float(inputPar['<Cshape>'][2]),
                          float(inputPar['<Cshape>'][3]),
                          float(inputPar['<Cshape>'][4]))
        self.normaliser= ('Vanadium', inputPar['<Nshape>'][1],
                         float(inputPar['<Nshape>'][2]),
                         float(inputPar['<Nshape>'][3]),
                         float(inputPar['<Nshape>'][4]))
        self.absorber = (inputPar['<Amater>'][1], inputPar['<Ashape>'][1],
                         float(inputPar['<Ashape>'][2]),
                         float(inputPar['<Ashape>'][3]),
                         float(inputPar['<Ashape>'][4]))
        self.environ = (inputPar['<Emater>'][1], inputPar['<Eshape>'][1],
                            float(inputPar['<Eshape>'][2]),
                            float(inputPar['<Eshape>'][3]),
                            float(inputPar['<Eshape>'][4]))
        self.Description = inputPar['<Sdescr>'][1]             # Sample description
        self.TempK = float(inputPar['<StempK>'][1])            # Sample temperature (in K)
        self.TempC = float(inputPar['<StempK>'][1])-273.15     # Sample temperature (in C)
        self.NbrElements = int(inputPar['<SNelem>'][1])        # Sample number of elements
        self.Mass = float(inputPar['<Smassc>'][1])             # Sample mass in the container
        self.Density = float(inputPar['<Sdegcc>'][1])          # Sample density (in g/cm3)
        self.Shape = (inputPar['<Sshape>'][1],                 # Sample shape
                      float(inputPar['<Sshape>'][2]),          # Sample inner diameter
                      float(inputPar['<Sshape>'][3]),          # Sample outer diameter
                      float(inputPar['<Sshape>'][4]))          # Sample height
        self.Fullness = float(inputPar['<Sfulln>'][1])         # Sample fullness
        self.Height = self.Shape[3]                            # Sample height
#        print ('Sample shape --> ',self.Shape)
#        print ('inputPar --> ',inputPar["<Sshape>"])
        self.InnerDiam = self.Shape[1]
#        print('inner Diam ',self.InnerDiam)
        self.OuterDiam = self.Shape[2]
#        print('outer Diam ',self.OuterDiam)
        self.InnerRadius = self.InnerDiam/2.0
        self.OuterRadius = self.OuterDiam/2.0

        # This is just a check to avoid mistyping
        if float(self.container[2]) != self.OuterDiam:
            print('WARNING! Container inner diameter should be equal',
                           'to sample outer diameter')
            msg = '    Container id = {} mm != Sample od = {} mm'
            print (msg.format(float(self.container[2]),self.OuterDiam))
        self.Title = inputPar['<Stitle>'][1]

        self.Volume = (np.pi * self.Height / 1000.0 *
                       (self.OuterRadius-self.InnerRadius)**2)
        self.EffDensity = self.Mass / self.Volume
        self.PackingFraction = self.EffDensity / self.Density

        self.conFile   = inputPar['<CoFile>'][1].strip()
        self.norFile   = inputPar['<NoFile>'][1].strip()
        self.envFile   = inputPar['<EnFile>'][1].strip()
        self.File      = inputPar['<SaFile>'][1].strip()
        self.absFile   = inputPar['<AbFile>'][1].strip()

# Reading scattering data from the 3-column ASCII files
#        print('Opening',self.File)
        # self.sampleData      = read_xye(self.sampleFile)
        # self.environmentData = read_xye(self.environmentFile)
        # self.containerData   = read_xye(self.containerFile)
        # self.vanadiumData    = read_xye(self.vanadiumFile)
        # self.absorberData    = read_xye(self.absorberFile)
        self.Data      = read_3col(self.File)
        self.envData   = read_3col(self.envFile)
        self.conData   = read_3col(self.conFile)
        self.norData   = read_3col(self.norFile)
        self.absData   = read_3col(self.absFile)


        atoms = {}
        natoms = 0.0
        for elem in range(self.NbrElements):
            ordatom = str(elem+1).zfill(2)
            label = '<Sato'+ordatom+'>'
#            print (elem,ordatom,inputPar[label][0],inputPar[label][1],inputPar[label][2])
#            print (elem,ordatom,inputPar[label])
            natoms += float(inputPar[label][2])
            atoms[inputPar[label][1]] = float(inputPar[label][2])
#            symbols[inputPar[label][1]] = inputPar[label][2]
#        print (atoms)
        self.natoms = natoms
        self.atoms = atoms
        self.symbols = list(atoms.keys())
# Calculating the concentration of each element in the sample
        conc = {}
        for elem in list(atoms.keys()):
#            print (elem,ordatom,inputPar[label][0],inputPar[label][1],inputPar[label][2])
            conc[elem] = atoms[elem]/natoms
#        print (atoms)
        self.conc = conc

        self.nuclei = {}
        for elem in list(atoms.keys()):
            self.nuclei[elem] = (elem, atoms[elem],conc[elem],elemento(elem).weight,
                                 elemento(elem).bcoh,elemento(elem).binc,
                                 elemento(elem).sig_coh,elemento(elem).sig_inc,
                                 elemento(elem).sig_sca,elemento(elem).sig_abs)

        self.molarMass = 0
        self.bcoh = 0
        self.binc = 0
        self.scoh = 0
        self.sinc = 0
        self.ssca = 0
        self.sabs = 0
        for elem in list(atoms.keys()):
            self.molarMass += elemento(elem).weight * conc[elem]
            self.bcoh += elemento(elem).bcoh    * conc[elem]
            self.binc += elemento(elem).binc    * conc[elem]
            self.scoh += elemento(elem).sig_coh * conc[elem]
            self.sinc += elemento(elem).sig_inc * conc[elem]
            self.ssca += elemento(elem).sig_sca * conc[elem]
            self.sabs += elemento(elem).sig_abs * conc[elem]
#        self.molarMass = molarMass
        self.bcoh = cmath.polar(self.bcoh)[0]
        self.binc = cmath.polar(self.binc)[0]
        self.free = self.ssca * (self.molarMass/(self.molarMass+elemento('n1').weight))**2
        self.b2lowQ = self.ssca/4/np.pi
        self.b2highQ = self.free/4/np.pi
        self.sabswl = getAbsXS(self.sabs,self.wavelength)

        self.AtomicDensity = getAtomicDensity(density=self.Density,
                                              molarMass=self.molarMass)
        self.EffAtomicDensity = getAtomicDensity(density=self.EffDensity,
                                                 molarMass=self.molarMass)


    def showTable(self,outfile=None):
        if outfile:
        # Redirect the output to a given file
            sys.stdout = open(outfile, 'w')

        print (120*'-')
        print ('Input parameters from {}'.format(self.inputPar['<par>'][1]))
        print (''.format())
        print (' {:<15} {:<15} {:<20} {:<20} {:<25} {:<20}'.format(
            *self.inputPar['<ins>'],
            *self.inputPar['<cyc>'],
            *self.inputPar['<env>']))
        print (' {:<15} {:<15} {:<20} {:<20} {:<25} {:<20}'.format(
            *self.inputPar['<pro>'],
            *self.inputPar['<mpr>'],
            *self.inputPar['<lco>']))
        print (' {:<15} {:<15} {:<20} {:<20} {:<25} {:<20}'.format(
            *self.inputPar['<sta>'],
            *self.inputPar['<nbk>'],
            *self.inputPar['<pag>']))
        print (' {:<15} {:<15} {:<20} {:<20} {:<25} {:<15}'.format(
            *self.inputPar['<wle>'],
            *self.inputPar['<zac>'],
            *self.inputPar['<apr>']))
        print (' {}'.format('Beam dimensions:'))
        print (' {:<9} {:<5} {:<12} {:<5} | {:<7} {:<5} | {:<10} {:<5} {:<11} {:<5} | {:<7} {:<5}'.format(
            *self.inputPar['<top>'],
            *self.inputPar['<bot>'],
            'Height (mm)=  ',self.beamHeight,
            *self.inputPar['<lft>'],
            *self.inputPar['<rgt>'],
            'Width (mm)=  ',self.beamWidth))
        print (''.format())
        print (' {}'.format('Sample dimensions:'))
        print (' {:<22} {:<6} {:<20} {:<12} {:>6} mm {:>6} mm {:>6} mm'.format(
            *self.inputPar['<Cmater>'],
            *self.inputPar['<Cshape>']))
        print (' {:<22} {:<6} {:<20} {:<12} {:>6} mm {:>6} mm {:>6} mm'.format(
            *self.inputPar['<Amater>'],
            *self.inputPar['<Ashape>']))
        print (' {:<22} {:<6} {:<20} {:<12} {:>6} mm {:>6} mm {:>6} mm'.format(
            *self.inputPar['<Nmater>'],
            *self.inputPar['<Nshape>']))
        print (' {:<22} {:<6} {:<20} {:<12} {:>6} mm {:>6} mm {:>6} mm'.format(
            *self.inputPar['<Emater>'],
            *self.inputPar['<Eshape>']))
        print ('{:<22} {:<6}  {:<20} {:<12} {:>6} mm {:>6} mm {:>6} mm'.format(
            '','',*self.inputPar['<Sshape>']))

        print (''.format())
        print (' {:<22} {:<15}'.format(
            *self.inputPar['<Sdescr>']))
        print (' {:<22} {:>10} K          {:<25} {:>10} g          {:<15} {:>10} cm3'.format(
            *self.inputPar['<StempK>'],
            *self.inputPar['<Smassc>'],
            'Sample volume=  ',format(self.Volume,'.6f')))
        print (' {:<22} {:>10} g/cm3      {:<25} {:>10} g/cm3'.format(
            *self.inputPar['<Sdegcc>'],
            'Effective density=  ',format(self.EffDensity,'.6f')))
        print (' {:<22} {:>10} units/Å3   {:<25} {:>8} units/Å3'.format(
            'Atomic density=  ',format(self.AtomicDensity,'.6f'),
            'Effective atomic density=  ',format(self.EffAtomicDensity,'.6f')))
        print (' {:<22} {:>10}            {:<25} {:>10}'.format(
            'Packing fraction=  ',format(self.PackingFraction,'.6f'),
            *self.inputPar['<Sfulln>']))
        print (' {:<22} {:>10} per unit   {:<25} {:>10} per unit'.format(
            *self.inputPar['<SNelem>'],'Number of atoms=',self.natoms))
        print (''.format())

        # Header of the table of elements in the sample
        print("-" * (140))
        print("{:^10} | {:^10} | {:^15} | {:^10} | {:^14} | {:^14} | {:^10} | {:^10} | {:^10} | {:^10}"
              .format("Nucleus","Atoms","Concentration","Molar mass","bcoh","binc","scoh","sinc","ssca","sabs"))
        print("{:^10} | {:^10} | {:^15} | {:^10} | {:^14} | {:^14} | {:^10} | {:^10} | {:^10} | {:^10}"
              .format(" "," "," ","g/mole","fm","fm","barns","barns","barns","barns"))
        print("-" * (140))
        for elem in list(self.atoms.keys()):
            print("{:^10} | {:^10.1f} | {:^15.6f} | {:^10.4f} | {:^14.3f} | {:^14.3f} | {:^10.5f} | {:^10.5f} | {:^10.5f} | {:^10.5f}"
                  .format(*self.nuclei[elem]))
        print("-" * (140))

        # Header of the sample properties
        print("-" * (130))
        print(" {:^10} | {:^10} | {:^10} | {:^10} | {:^10} | {:^10} | {:^10} | {:^10} | {:^10} | {:^10}"
              .format("Molar mass","bcoh","binc","scoh","sinc","ssca","sabs","sabs(wl)","<b>2","<b_free>2"))
        print(" {:^10} | {:^10} | {:^10} | {:^10} | {:^10} | {:^10} | {:^10} | {:^10} | {:^10} | {:^10}"
              .format("g/mole","fm","fm","barns","barns","barns","barns","barns","b/sr/u","b/sr/u"))
        print("-" * (130))
        print(" {:^10.4f} | {:^10.3f} | {:^10.3f} | {:^10.5f} | {:^10.5f} | {:^10.5f} | {:^10.5f} | {:^10.5f} | {:^10.5f} | {:^10.5f}"
            .format(self.molarMass,self.bcoh,self.binc,self.scoh,self.sinc,self.ssca,self.sabs,self.sabswl,self.b2lowQ,self.b2highQ))
        print("-" * (130))

        if outfile:
            # Reset the standard output to the default (the screen)
            sys.stdout = sys.__stdout__

#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def execute_command_and_write_to_file(command, output_file):
    try:
        # Open the file in write mode
        with open(output_file, 'w') as file:
            # Run the command and redirect the output to the file
            result = subprocess.run(command, check=True, text=True, stdout=file, stderr=subprocess.PIPE)

            # Access the return code
            return_code = result.returncode

            # Print the return code
            print("Return Code:", return_code)

    except subprocess.CalledProcessError as e:
        # Handle errors (non-zero exit code)
        print("Error:", e)
        print("Command failed with return code", e.returncode)
        print("Error output:", e.stderr)

#--------1---------2---------3---------4---------5---------6---------7---------8


#--------1---------2---------3---------4---------5---------6---------7---------8

def generate_com(CORRECT_template="correct_template.com",name="prueba",
                 rsample=0.5, density=0.11, MTname="MTBelljar_6mm",
                 MTCanname='MTCan_6mm', rcan=0, Vananame="Vanadium_7mm",
                 rvana=0.7, beam_width=0.6, beam_height=1.6):
    """
    Generate an input file for CORRECT starting from a template, then run CORRECT.

    The units of these two distances must be the same, and the result is given
    in the same unit3 (if distances are in mm, the volume will be in mm3).

    Parameters
    ----------
    - name : string, optional
    - rsample : float, optional
    - density : float, optional
        The height of the cylinder. The default is 50.0.

    Returns
    -------
    float : The volume of the cylinder.

    Author: Jose Robledo
    Date: Dec 7 2022
--------------------------------------------------------------------------------
    """
    with open(CORRECT_template, "r") as f:
        lines = f.readlines()
        text = "".join(lines)

    text = text.replace("{pyfname}",name)
    text = text.replace("{pyradius_cm}", str(rsample))
    text = text.replace("{pydensity_at_AA3}",str(density))
    text = text.replace("{pymtbelljarname}", MTname)
    text = text.replace("{pycanname}",MTCanname)
    text = text.replace("{pyradius_can}",str(rcan))
    text = text.replace("{pyradius_vana}",str(rvana))
    text = text.replace("{pyvanadiumname}", Vananame)
    text = text.replace("{pybeam_w_cm}",str(beam_width))
    text = text.replace("{pybeam_h_cm}",str(beam_height))
    print(text)

    with open(f"{name}.com", "w") as f:
        f.writelines(text)

#    run_cmd(f"/nethome/dif/d4soft/bin/correct @{name}", stdout=f"/net4/serdon/illdata/233/d4/exp_6-05-1067/processed/oski/output_corr_{name}.txt")

    executable_name = "/nethome/dif/d4soft/bin/correct"
    arg1 = f"@{name}"
    # Example of usage
    command_to_run = ["executable_name", "arg1"]
    output_file_path = "output.txt"

    execute_command_and_write_to_file(command_to_run, output_file_path)

    sys.stdout = sys.__stdout__

#--------1---------2---------3---------4---------5---------6---------7---------8


#--------1---------2---------3---------4---------5---------6---------7---------8
def saveCORRECT(expt):
    parfile = expt.sampleTitle+".com"
    with open(parfile, "w") as f:
        f.write('! '+parfile+'\n')
        f.write('!\n')
        f.write('instr '+expt.instrument+'\n')
        line=(
            'sample "{}.adat" {} /temperature={} /density={} /packing={:8.6f} /fullness={}'.
            format(expt.sampleTitle, expt.sampleOuterRadius/10.0,
                expt.sampleTemperature, expt.sampleDensity,
                expt.samplePackingFraction,expt.sampleFullness))
        f.write(line+'\n')
        # for i in range(expt.sampleNbrElements):
        #     keyatom = '<Sato'+str(i+1).zfill(2)+'>'
        #     f.write('component {}\n'.format(inputPar[keyatom][1]))
        #     print('component {}'.format(inputPar[keyatom][1]))
        can = expt.containerFile[0:-5]+'.adat'
        f.write('container "{}" {}\n'.format(can,expt.container[3]/20.0))
        bckg = expt.environmentFile[0:-5]+'.adat'
        f.write('background "{}" 0.8\n'.format(bckg))
        f.write('! black "absorber.adat" 0.93\n')
        vana = expt.vanadiumFile[0:-5]+'.adat'
        f.write('vanadium "{}" {} /smoothing=1 /multiplier=1.02\n'.
                format(vana,expt.vanadium[3]/20.0))
        f.write('background /vanadium "{}" 0.85\n'.format(bckg))
        f.write('wavelenght {}\n'.format(expt.wavelength))
        f.write('! zeroangle = {} already subtracted\n'.format(expt.zeroAngle))
        f.write('zeroangle {}\n'.format(0.0))
        f.write('beam {} {}\n'.format(expt.beamHeight/10.0,expt.beamWidth/10.0))
        f.write('! xout angle\n')
        f.write('! output '+expt.sampleTitle+'.corr\n')
        f.write('! title "'+expt.sampleTitle+'.corr (after correct)"\n')
        f.write('xout q\n')
        f.write('output '+expt.sampleTitle+'.corr.q\n')
        f.write('title "'+expt.sampleTitle+'.corr.q (after correct)"\n')
        f.write('spectrum 1\n')
        f.write('execute/nopause\n')
        f.write('quit\n')
    return

#--------1---------2---------3---------4---------5---------6---------7--------8
def saveFile_xye(filename,x,y,e,heading):
    """
saveFile_3col function

This function creates a 3 column ASCII file, with x,y,error.

As first line of the heading, it writes the filename preceded by "# "
Then, it prints as many lines as elements contains the list heading.
Finally, it writes a line for each point with (x,y,error)

Input:
    - filename: a string containing the output filename
    - x,y,e: 3 lists with the same number of elements, containing abcissa, ordinate and error
    - heading: A list where each element will be a line in the heading of the output file

Use:
    saveFile_3col(filename,x,y,e,heading)

Created on Thu Oct 19 2021
@author: Gabriel Cuello
---------------------------------------
    """
    with open(filename,'w') as datafile:
        datafile.write("# "+filename+'\n')
        for i in range(len(heading)):
            datafile.write("# "+heading[i]+'\n')
        for i in range(len(x)):
            datafile.write("{: 9.3f}".format(x[i])+' '+
                           "{:18.6f}".format(y[i])+' '+
                           "{:18.6f}".format(e[i])+'\n')
        print('File '+filename+' saved')
    return
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7--------8
def saveFile_3col(filename,data,heading):
    """
saveFile_3col function

This function creates a 3 column ASCII file, with x,y,error.

As first line of the heading, it writes the filename preceded by "# "
Then, it prints as many lines as elements contains the list heading.
Finally, it writes a line for each point with (x,y,error)

Input:
    - filename: a string containing the output filename
    - x,y,e: 3 lists with the same number of elements, containing abcissa, ordinate and error
    - heading: A list where each element will be a line in the heading of the output file

Use:
    saveFile_3col(filename,x,y,e,heading)

Created on Thu Oct 19 2021
@author: Gabriel Cuello
---------------------------------------
    """
    x = data[:,0]
    y = data[:,1]
    e = data[:,2]
    with open(filename,'w') as datafile:
        datafile.write("# "+filename+'\n')
        for i in range(len(heading)):
            datafile.write("# "+heading[i]+'\n')
        for i in range(len(x)):
            datafile.write("{: 9.3f}".format(x[i])+' '+
                           "{:18.6f}".format(y[i])+' '+
                           "{:18.6f}".format(e[i])+'\n')
        print('File '+filename+' saved')
    return
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def saveCorrelations(filename,ar,pcf,pdf,rdf,tor,run,heading):
    """
saveCorrelations function

This function creates a 6 column ASCII file, with the real space correlation functions

As first line of the heading, it writes the filename preceded by "# "
Then, it prints as many lines as elements contains the list heading.
Finally, it writes a line for each point with r,pcf,pdf,rdf,tor,run

Input:
    - filename: a string containing the output filename
    - ar, pcf, pdf, rdf, tor, run: 6 lists with the same number of elements, containing r and the functions
    - heading: A list where each element will be a line in the heading of the output file

Use:
    saveCorrelations(filename, ar, pcf, pdf, rdf, tor, run, heading)

Created on Mon Oct 30 2023
Midified 15/05/2024
@author: Gabriel Cuello
---------------------------------------
    """
    with open(filename, 'w') as datafile:
        datafile.write("# " + filename + '\n')
        datafile.writelines("# " + line + '\n' for line in heading)
        for vals in zip(ar, pcf, pdf, rdf, tor, run):
            datafile.write("{:9.3f} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f}\n".format(*vals))
    print('File ' + filename + ' saved')
    return
#--------1---------2---------3---------4---------5---------6---------7---------



#--------1---------2---------3---------4---------5---------6---------7---------
def saveRSCF(filename,fou,heading):
    """
saveRSF function

This function creates a 6 column ASCII file, with the real space correlation functions

As first line of the heading, it writes the filename preceded by "# "
Then, it prints as many lines as elements contains the list heading.
Finally, it writes a line for each point with (x,y,error)

Input:
    - filename: a string containing the output filename
    - x,y1,y2,y3,y4,y5: 6 lists with the same number of elements, containing r and the functions
    - heading: A list where each element will be a line in the heading of the output file

Use:
    saveRSCF(filename,x,y1,y2,y3,y4,y5,heading)

Created on Mon Dec 13 2021
@author: Gabriel Cuello
---------------------------------------
    """
    with open(filename,'w') as datafile:
        datafile.write("# "+filename+'\n')
        for i in range(len(heading)):
            datafile.write("# "+heading[i]+'\n')
        for i in range(len(fou)):
            datafile.write("{: 9.3f}".format(fou[i][0])+' '+
                           "{:12.6f}".format(fou[i][1])+' '+
                           "{:12.6f}".format(fou[i][2])+' '+
                           "{:12.6f}".format(fou[i][3])+' '+
                           "{:12.6f}".format(fou[i][4])+' '+
                           "{:12.6f}".format(fou[i][5])+'\n')
        print('File '+filename+' saved')
    return
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def read_xye(filename):
    '''
read_3col function

Opens a file and read it line by line. This function assumes it is a 3-columns file.
The lines containig the symbol # are ignored. Be careful, # could be anywhere in the line!
The empty lines are also ignored.
As output it produces 3 lists with abcissas, ordinates and errors for the ordinates.

Use:
    x_dat, y_dat, e_dat = readD4_3col('mydatafile.dat')

Created on Wed Dec 30, 2020
@author: Gabriel Cuello
---------------------------------------
    '''
    data = open(filename,'r') # Opens the data file in read only mode

    # Creating the lists that will contain abcissas, ordinates and errors
    x = []
    y = []
    e = []

    # Reading the file line by line
    for dataline in data.readlines():
    # dataline is a string that contains each line of the file in data.
    # note that the last character of the string is a 'carriage return', \n
        if '#' not in dataline:  # Only the lines without # are treated.
            # the method .strip(' ') removes blanks at the beginning of the string
            row = dataline.strip(' ')[:-1]
            if len(row)>0:  # Only the no-empty lines are treated
                columns = row.split()   # This method split the line using the spaces
                x.append(float(columns[0]))
                y.append(float(columns[1]))
                if (len(columns)==3):
                    if (columns[2] == 'i') or (columns[2] == 'o'):
                        e.append(float(0.0))
                    else:
                        e.append(float(columns[2]))
                else:
                    e.append(float(0.0))
    data.close()
    print ('The data file {} read with no errors. Number of data = {}'.
           format(filename,len(x)))
#    return np.array(x,y,e)
#   Converts the lists in arrays
    xa=np.array(x)
    ya=np.array(y)
    ea=np.array(e)
    return xa,ya,ea
# End of read_xye
#--------1---------2---------3---------4---------5---------6---------7---------

#--------1---------2---------3---------4---------5---------6---------7---------
def read_3col(filename):
    '''
read_3col function

Opens a file and read it line by line. This function assumes it is a 3-columns file.
The lines containig the symbol # are ignored. Be careful, # could be anywhere in the line!
The empty lines are also ignored.
As output it produces 3 lists with abcissas, ordinates and errors for the ordinates.

Use:
    x_dat, y_dat, e_dat = readD4_3col('mydatafile.dat')

Created on Wed Dec 30, 2020
@author: Gabriel Cuello
---------------------------------------
    '''
    data = open(filename,'r') # Opens the data file in read only mode

    # Creating the lists that will contain abcissas, ordinates and errors
    x = []
    y = []
    e = []

    # Reading the file line by line
    for dataline in data.readlines():
    # dataline is a string that contains each line of the file in data.
    # note that the last character of the string is a 'carriage return', \n
        if '#' not in dataline:  # Only the lines without # are treated.
            # the method .strip(' ') removes blanks at the beginning of the string
            row = dataline.strip(' ')[:-1]
            if len(row)>0:  # Only the no-empty lines are treated
                columns = row.split()   # This method split the line using the spaces
                x.append(float(columns[0]))
                y.append(float(columns[1]))
                if (len(columns)==3):
                    if (columns[2] == 'i') or (columns[2] == 'o'):
                        e.append(float(0.0))
                    else:
                        e.append(float(columns[2]))
                else:
                    e.append(float(0.0))
    data.close()
    print ('The data file {} read with no errors. Number of data = {}'.
           format(filename,len(x)))
#    return np.array(x,y,e)
#   Converts the lists in arrays
    xa=np.array(x)
    ya=np.array(y)
    ea=np.array(e)
#   Reshapes the arrays as 2D arrays, but with 1 column
    xa=xa.reshape(xa.shape[0],1)
    ya=ya.reshape(ya.shape[0],1)
    ea=ea.reshape(ea.shape[0],1)
#   Concatenates the arrays to have a 3-column matrix
    data = np.concatenate((xa,ya),axis=1)
    data = np.concatenate((data,ea),axis=1)
    return data
# End of read_3col
#--------1---------2---------3---------4---------5---------6---------7---------

class DataXYE:
    def __init__(self, filename):
        self.data = np.loadtxt(filename)
        self.filename = filename
        self.title = "Data in "+filename
        self.x = self.data[:, 0]
        self.y = self.data[:, 1]
        self.e = self.data[:, 2]
        self.scale_type = "a"

    def plot(self, scale_type=None, labelx="X-axis", labely="Y-axis", title=None, color="blue",
             curve_type="-", min_x=None, min_y=None, max_x=None, max_y=None):
#        plt.figure(figsize=(9, 6))
        self.fig, ax = plt.subplots()
        scale_type = scale_type if scale_type is not None else self.scale_type
        title = title if title is not None else self.title


        ax.set_ylabel("Intensity (arb. units)")
        if scale_type == "Q" or scale_type == "q":
            ax.set_xlabel(r'$Q$ (Å$^{-1}$)')
        elif scale_type == "A" or scale_type == "a":
            ax.set_xlabel(r'$2\theta$ (˚)')
        elif scale_type == "R" or scale_type == "r":
            ax.set_xlabel('$R$ (Å)')
        else:
            plt.xlabel(labelx)
            plt.ylabel(labely)

        ax.set_title(title)

        if min_x is not None:
            ax.set_xlim(left=min_x)
        if max_x is not None:
            ax.set_xlim(right=max_x)
        if min_y is not None:
            ax.set_ylim(bottom=min_y,top=max_y)
        if max_y is not None:
            ax.set_ylim(top=max_y)

        ax.plot(self.data[:, 0], self.data[:, 1], label=self.filename, color=color, linestyle=curve_type)

        ax.grid(True)
        ax.legend()

        plt.show()

# # Example usage:
# filename = "your_data_file.txt"  # Replace with your actual data file
# data_object = DataXYE(filename)

# # Customize and plot the data
# data_object.plot(scale_type="log", labelx="Time", labely="Value", title="Customized Plot",
#                   color="red", curve_type="--", min_x=0.1, min_y=1e-3, figure_name="custom_figure")

# # Show the plot
# data_object.show_plot()



class DataXYE2:
    def __init__(self, file):
        self.data = np.loadtxt(file)
        self.file = file
        self.title = "Plotting "+self.file
        self.labelx = "Axis X"
        self.labely = "Axis Y"
        self.color = "blue"
        self.curve = "-"
        self.scale = "a"
        self.superplot = False

#         self.title = title
#         self.labelx = labelx
#         self.labely = labely

#         self.color = color
#         self.curve = curve

    def plot(self, scale="a", superplot=False, title=None, labelx=None, labely=None, color=None, curve=None):
        title = title if title is not None else self.title
        labelx = labelx if labelx is not None else self.labelx
        labely = labely if labely is not None else self.labely
        color = color if color is not None else self.color
        curve = curve if curve is not None else self.curve

        if not superplot: plt.figure(figsize=(9, 6))

        plt.plot(self.data[:, 0], self.data[:, 1], color=color, linestyle=curve, label=self.filename)

        plt.legend(loc='best')
        plt.title('Data in ' + self.file)
        plt.ylabel('Intensity (arb. units)')
        if scale == "Q" or self.scale == "q":
            plt.xlabel(r'$Q$ (Å${-1}$)')
        elif scale == "A" or self.scale == "a":
            plt.xlabel(r'$2\theta$ (˚)')
        elif scale == "R" or self.scale == "r":
            plt.xlabel('$R$ (Å)')
        else:
            plt.xlabel(labelx)
            plt.ylabel(labely)

        # Configurar la escala de los ejes
        x_range = np.ptp(self.data[:, 0])  # Rango del eje X
        y_range = np.ptp(self.data[:, 1])  # Rango del eje Y

        plt.axis([np.min(self.data[:, 0]) - 0.1 * x_range, np.max(self.data[:, 0]) + 0.1 * x_range,
                  np.min(self.data[:, 1]) - 0.1 * y_range, np.max(self.data[:, 1]) + 0.1 * y_range])
        plt.grid(True)
        plt.tight_layout()

        if not superplot: plt.show()




#         title = title if title is not None else self.title
#         labelx = labelx if labelx is not None else self.labelx
#         labely = labely if labely is not None else self.labely
#         color = color if color is not None else self.color
#         curve = curve if curve is not None else self.curve
#         fig, ax = plt.subplots()
#         ax.plot(self.data[:, 0], self.data[:, 1], label=self.file, color=color, linestyle=curve)
#         ax.set_title(self.title)
#         ax.set_xlabel(labelx)
#         ax.set_ylabel(labely)


#         # Configurar la escala de los ejes
#         x_range = np.ptp(self.data[:, 0])  # Rango del eje X
#         y_range = np.ptp(self.data[:, 1])  # Rango del eje Y
#         ax.set_xlim([np.min(self.data[:, 0]) - 0.1 * x_range, np.max(self.data[:, 0]) + 0.1 * x_range])
#         ax.set_ylim([np.min(self.data[:, 1]) - 0.1 * y_range, np.max(self.data[:, 1]) + 0.1 * y_range])

#         ax.legend()
#         plt.show()

# # Ejemplo de uso
# archivo_datos = "tu_archivo.txt"  # Reemplaza con el nombre de tu archivo de datos
# graficador = GraficadorDatos(archivo_datos)

# # Configurar opciones (opcional)
# graficador.configurar_titulo("Mi Gráfico")
# graficador.configurar_leyendas("Tiempo", "Valor")
# graficador.configurar_estilo("red", "--")

# # Graficar
# graficador.graficar()





class DataXYE0:

    def __init__(self, filename, symbol="r-+", save=False, scale="a", superplot=False, xmin=None, xmax=None, ymin=None, ymax=None):
        self.data = np.loadtxt(filename)
        self.filename = filename
        self.ext = os.path.splitext(filename)[1][1:]
        self.x = self.data[:, 0]
        self.y = self.data[:, 1]
        self.e = self.data[:, 2]

        self.xave = st.mean(self.x)
        self.xmin = min(self.x)
        self.xmax = max(self.x)
        self.yave = st.mean(self.y)
        self.ymin = min(self.y)
        self.ymax = max(self.y)
        self.eave = st.mean(self.e)
        self.emin = min(self.e)
        self.emax = max(self.e)

        self.scale = scale
        self.symbol = symbol
        self.minx = xmin
        self.maxx = xmax
        self.miny = ymin
        self.maxy = ymax

        xmin = xmin if xmin is not None else self.minx
        xmax = xmax if xmax is not None else self.maxx
        ymin = ymin if ymin is not None else self.miny
        ymax = ymax if ymax is not None else self.maxy



    def plot(self, symbol, superplot=False, xmin=None, xmax=None, ymin=None, ymax=None):

        if not superplot: plt.figure(figsize=(9, 6))
        print('***',self.symbol)
        plt.plot(self.x, self.y, self.symbol, label=self.filename)

        plt.legend(loc='best')
        plt.title('Data in ' + self.filename)
        if self.scale == "Q" or self.scale == "q":
            plt.xlabel(r'$Q$ (Å${-1}$)')
        elif self.scale == "A" or self.scale == "a":
            plt.xlabel(r'$2\theta$ (˚)')
        elif self.scale == "R" or self.scale == "r":
            plt.xlabel('$R$ (Å)')
        else:
            plt.xlabel('Abscissa')
        plt.ylabel('Intensity (arb. units)')
        plt.axis([xmin, xmax, ymin, ymax])
        plt.grid(True)
        plt.tight_layout()

        if not superplot: plt.show()


    def show(self):
        for i in range(len(self.x)):
            print(f"{self.x[i]:>6}{self.y[i]:>12}{self.e[i]:>12}")



#--------1---------2---------3---------4---------5---------6---------7--------8
    def save(self,filename):
        """
saveFile_3col function

This function creates a 3 column ASCII file, with x,y,error.

As first line of the heading, it writes the filename preceded by "# "
Then, it prints as many lines as elements contains the list heading.
Finally, it writes a line for each point with (x,y,error)

Input:
    - filename: a string containing the output filename
    - x,y,e: 3 lists with the same number of elements, containing abcissa, ordinate and error
    - heading: A list where each element will be a line in the heading of the output file

Use:
    saveFile_3col(filename,x,y,e,heading)

Created on Thu Oct 19 2021
@author: Gabriel Cuello
---------------------------------------
        """
        x = self.data[:,0]
        y = self.data[:,1]
        e = self.data[:,2]
        with open(filename,'w') as datafile:
            for i in range(len(x)):
                datafile.write("{: 9.3f}".format(x[i])+' '+
                               "{:18.6f}".format(y[i])+' '+
                               "{:18.6f}".format(e[i])+'\n')
            print('File '+filename+' saved')
        return
#--------1---------2---------3---------4---------5---------6---------7---------



#--------1---------2---------3---------4---------5---------6---------7---------
class DataXYE_old():
    """
Type: Class

Object:
    To read data files in ASCII format, two (X Y) or three (X Y E) columns

Input:
    filename: (string) Filename of the file containing the data.

Output:
    An instance created with the following attributes and methods.

    self.filename: (string) Filename of the input file
    self.basename: (string) File basename (what it is before extention's dot)
    self.ext: (string) File extension (without the dot)
    self.x: Abscissas
    self.y: Ordinates
    self.e: Errors (or 3rd coordinate). Returns -1 for 2-column files.
    self.head: List of strings with each line in the file header.

    self.xave: Mean value of the abscissas
    self.yave: Mean value of the ordinates
    self.eave: Mean value of the errors

    self.xmin: Minimum value of the abscissas
    self.ymin: Minimum value of the ordinates
    self.emin: Minimum value of the errors

    self.xmax: Maximum value of the abscissas
    self.ymax: Maximum value of the ordinates
    self.emax: Maximum value of the errors

    self.plot(): Makes a simple plot of y coordinate
    self.show(): Shows the data on screen (as a 3-column table)
    self.header(): Prints the header of the file

Author: Gabriel Cuello
Created: 29/12/2022
Modified:
#--------1---------2---------3---------4---------5---------6---------7---------
    """

    def __init__(self,filename):
        """
Type: Main function of the Class DataXYE
    The file is read and the attributes are defined here.

Input:
    filename: (string) The filename of the file containing the data.

Output:
    The attributes that can be accessed by the instances.
    See the help of this Class for a complete list of attributes.

Author: Gabriel Cuello
Created: 29/12/2022
Modified:
#--------1---------2---------3---------4---------5---------6---------7---------
        """
        self.filename = filename
        self.basename = os.path.splitext(filename)[0]
        self.ext = os.path.splitext(filename)[1][1:] # Exclude 1st character to avoid the dot
        self.x = []
        self.y = []
        self.e = []
        self.head = []

        data = open(self.filename,'r')
        lines = data.readlines()
        for dataline in lines:
            row = dataline.strip(' ')[:-1]
            if len(row)>0:  # Only the non empty lines are treated
                if row[0] == "#" or row[0] == "!":
                    self.head.append(row)
                else:
                    columns = row.split()   # This method split the line using the spaces
                    if len(columns) == 2:
                        self.x.append(float(columns[0]))
                        self.y.append(float(columns[1]))
                        self.e.append(-1.0)
                    elif len(columns) == 3:
                        self.x.append(float(columns[0]))
                        self.y.append(float(columns[1]))
                        self.e.append(float(columns[2]))
                    else:
                        print ("Wrong file format")
                        sys.exit()
        data.close()
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.e = np.array(self.e)
        self.xave = st.mean(self.x)
        self.xmin = min(self.x)
        self.xmax = max(self.x)
        self.yave = st.mean(self.y)
        self.ymin = min(self.y)
        self.ymax = max(self.y)
        self.eave = st.mean(self.e)
        self.emin = min(self.e)
        self.emax = max(self.e)
        self.peaks_x, self.peaks_y = find_peaks_in_range(self.x, self.y, 5.0, 40.0)
        self.xminr, self.yminr = find_minimum_within_range(self.x, self.y, 5.0, 40.0)

    def plot(self,file_format=0,xmin=None,xmax=None, ymin=None, ymax=None):
        """
Type: Method in DataXYE class

Object:
    To make a simple plot of the ordinates as function of abscissas.
    To produce a file with the plot.

Input:
    xmin,xmax: Minimum and maximum values of the x-axis (float, optional)
    ymin,ymax: Minimum and maximum values of the y-axis (float, optional)
    file_format: A string that defines the format (and extension) of the ouput
                 file (string, optional)

Output:
    A simple plot on the screen.
    A file with the plot in a graphical file.

Remarks:
  * Several formats are possible for the output file. The kind of file is
    defined by the input parameter file_format, which can must take one
    of the following values: 'png','pdf','jpg','tiff','svg','jpeg','ps','eps'.
    If this paramteter is not present, it takes the default value 0 and no
    output file is created.

  * The output file has the same basename as the input file, but the extension
    corresponding to chosen format.

  * The limits of the axes are optional. Their default value is None, which
    will produce a plot with automatic limits.

Author: Gabriel Cuello
Created: 29/12/2022
Modified:
#--------1---------2---------3---------4---------5---------6---------7---------
        """
        A_ext = ['adat', 'Adat', 'reg']
        Q_ext = ['qdat', 'Qdat', 'Qreg', 'soq', 'SoQ']
        R_ext = ['pcf', 'pdf', 'tor', 'rdf', 'run']

        plt.figure(figsize=(9,6))

        plt.plot(self.x,self.y, 'r-+',label=self.filename)

        plt.legend(loc='best')
        plt.title('Data in ' + self.filename)
        plt.xlabel('Abscissa')
        if self.ext in A_ext:
            plt.xlabel(r'$2\theta$ (˚)')
        elif self.ext in Q_ext:
            plt.xlabel(r'$Q$ (Å${-1}$)')
        elif self.ext in R_ext:
            plt.xlabel('$R$ (Å)')
        plt.ylabel('Intensity (arb. units)')
        plt.axis([xmin, xmax, ymin, ymax])
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        if file_format in ['png','pdf','jpg','tiff','svg','jpeg','ps','eps']:
            file_fig = '../regdata/'+self.basename+'.'+file_format
            plt.savefig(file_fig, format=file_format)
            print("Figure saved on {}".format(file_fig))

    def show(self):
        """
Type: Method in DataXYE class

Object:
    To show the data on the screen.

Input: None

Output:
    Print out of data on the screen in a 3-column table.

Author: Gabriel Cuello
Created: 29/12/2022
Modified:
#--------1---------2---------3---------4---------5---------6---------7---------
        """
        A_ext = ['adat', 'Adat', 'reg']
        Q_ext = ['qdat', 'Qdat', 'Qreg', 'soq', 'SoQ']
        R_ext = ['pcf', 'pdf', 'tor', 'rdf', 'run']

        if self.ext in A_ext:
            print(f"{'Ang':>6}{'Intensity':>12}{'Error':>12}")
        elif self.ext in Q_ext:
            print(f"{'Q':>6}{'Intensity':>12}{'Error':>12}")
        elif self.ext in R_ext:
            print(f"{'R':>6}{'Intensity':>12}{'Ignore':>12}")
        else:
            print(f"{'x':>6}{'y':>12}{'e':>12}")
        for i in range(len(self.x)):
            print(f"{self.x[i]:>6}{self.y[i]:>12}{self.e[i]:>12}")

    def header(self,lines=1000):
        """
Type: Method in DataXYE class

Object:
    To print the file header on the screen.

Input:
    lines: number of lines to be printed

Output:
    Print out of file heading on the screen.

Remarks:
  * The default value for the input parameter lines is 1000. But this method
    will print only the number of lines in the header, unless this number is
    greater than 1000.

Author: Gabriel Cuello
Created: 29/12/2022
Modified:
#--------1---------2---------3---------4---------5---------6---------7---------
        """
        for i in range(min(lines,len(self.head))):
            print(f"{i+1:>3} {self.head[i]}")

#--------1---------2---------3---------4---------5---------6---------7---------


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#
#  Reading D4 raw data and regrouping diffractograms
#
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

#--------1---------2---------3---------4---------5---------6---------7---------8
def getDate():
    """
Produce a string with the current date in the format:
    Dayname DD/MM/YYYY HH:MM:SS

Input: nothing

Output: string, current_datetime

Author: Gabriel Cuello
Date:   Jan 2022
--------------------------------------------------------------------------------
    """
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    #    curr_date = datetime.today()
    #    day = ' '+calendar.day_name[curr_date.weekday()]+' '
    day = " " + calendar.day_name[now.weekday()] + " "
    current_datetime = day + now.strftime("%d/%m/%Y %H:%M:%S")
    return current_datetime
#--------1---------2---------3---------4---------5---------6---------7---------8


#--------1---------2---------3---------4---------5---------6---------7---------8
def getDTC_mcounts(monitor, time, dead_time=2.4e-6):
    """
Calculate the dead-time corrected monitor counts.

The measured counting rate is
        nu_meas = monitor / time
then, the dead-time corrected rate is
        nu = nu_meas / (1 - nu_meas * tau)
where tau is the dead-time in seconds and nu is given in counts/second.

Thus, the dead-time corrected counts are: nu * time.

Input: count, time, dead_time
    count : measured counts on the monitor
    time : measuring time (in seconds) (if negative, it changes to 1e-6)
    dead_time : monitor dead-time (in seconds) (default = 2.4e-6)

Output: Dead-time corrected monitor

Notes:
    (1) For D4, in 2021 the program d4creg uses tau=2.4E-06 s for the monitor.
    (2) The program d4creg uses the following expresion:
             nu = (1-sqrt(1-4*nu_meas*tau))/2/tau
    (3) In the Internal ILL Report (ILL98FI15T) by H.E. Fischer, P. Palleau and
        D. Feltin the tau for the monitor is 7.2E-06 s.
    (4) At some point Henry Fischer suggested using tau=12E-6 s for the monitor.

Author: Gabriel Cuello
Date:   Jan 2022
--------------------------------------------------------------------------------
    """
# If the input counting time is negative, it is forced to be 1e-6 sec, to avoid
# a division by 0.
    if time <= 0:
        time = 1e-6
    nu_meas = monitor / time  # counting rate
    correction = 1.0 - nu_meas * dead_time
    if correction > 0:
        factor = 1.0 / correction
        nu = nu_meas * factor

        with open('../logfiles/d4creg.log', 'a') as file:
            file.write("  Monitor DT = {:4.2e} s. Correction factor = {:10.6f} \n".format(
                dead_time, factor))

#        print(
#            "  Monitor DT = {:4.2e} s. Correction factor = {:10.6f}".format(
#                dead_time, factor))
    else:
        with open('../logfiles/d4creg.log', 'a') as file:
            file.write("--- ERROR in Function getDTC_mcounts: correction factor < 0 \n")
#        print("--- ERROR in Function getDTC_mcounts: correction factor < 0")
    return nu * time
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def getDTC_dcounts(counts, total_counts, time, dead_time=7e-06):
    """
Calculate the dead-time corrected detector counts.

The measured counting rate is
        nu_meas = counts / time
then, the dead-time corrected rate is
        nu = nu_meas / (1 - nu_meas * tau)
where tau is the dead-time in seconds and nu is given in counts/second.

Thus, the dead-time corrected counts are:  nu * time.

Input: count, time, dead_time
    count : measured counts on the detector (this a matrix, 9x64 or 9x128)
    time : measuring time (in seconds) (if negative, it changes to 1e-6)
    dead_time : monitor dead-time (in seconds) (default = 7e-6)

Output: Dead-time corrected detector counts (a matrix, 9x64 or 9x128)

Notes:
    (1) For D4, in 2021 the program d4creg uses tau=7E-06 for the microstrip
        detector.
    (2) The program d4creg uses the following expresion:
             nu = (1-sqrt(1-4*nu_m*tau))/2/tau
    (3) In the Internal ILL Report (ILL98FI15T) by H.E. Fischer, P. Palleau and
        D. Feltin the tau for the detector is 7.17E-06 s.
    (4) At some point Henry Fischer suggested using tau=0 s for the detector.

Author: Gabriel Cuello
Date:   Jan 2022
--------------------------------------------------------------------------------
    """
# If the input counting time is negative, it is forced to be 1e-6 sec, to avoid
# a division by 0.
    if time <= 0:
        time = 1e-6
    nu_meas = counts / time
    correction = 1.0 - nu_meas * dead_time
    if (correction.all()) > 0:
        factor = 1.0 / correction
        nu = nu_meas * factor
        with open('../logfiles/d4creg.log', 'a') as file:
            file.write("  Monitor DT = {:4.2e} s. Correction factor = {:10.6f}".format(
                dead_time, factor[0][0]) + " (1st detector cell)\n")
#        print(
#            "  Monitor DT = {:4.2e} s. Correction factor = {:10.6f}".format(
#                dead_time, factor[0][0]), "(1st detector cell")
    else:
        with open('../logfiles/d4creg.log', 'a') as file:
            file.write("--- ERROR in Function getDTC_dcounts: correction factor < 0 \n")
#        print("--- ERROR in Function getDTC_dcounts: correction factor < 0")
    return nu * time, total_counts * factor[0][0]
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def printRates(detector, counts, time, reactor_power=52):
    """
Print out the basic information about counting rates.

Input: detector, counts, time, reactor_power
    detector : string, 'detector' or 'monitor'
    counts : float, total number of counts on detectors or on the monitor
    time : float, counting time in seconds (if negative, it changes to 1e-6)
    reactor_power : float, reactor power in MW (default value = 52 MW)

Output: None

Author: Gabriel Cuello
Date:   Jan 2022
--------------------------------------------------------------------------------
    """
# If the input counting time is negative, it is forced to be 1e-6 sec, to avoid
# a division by 0.
    if time <= 0:
        time = 1e-6
    out = (
        "  Counts on {} = {:10.8g}  ===> counting-rate = {:10.8g} c/s".format(
            detector, counts, counts / time))
    with open('../logfiles/d4creg.log', 'a') as file:
         file.write(out + "\n")

#    print(out)
    if reactor_power <= 0:
        reactor_power = 52
    out = (
        "  Reactor-power normalised {} counting-rate = {:10.6g} c/s/MW".format(
            detector, counts / time / reactor_power))
    with open('../logfiles/d4creg.log', 'a') as file:
         file.write(out + "\n")
#    print(out)
    return
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def getRefLines(Lines):
    """
Determine the reference lines in the ASCII numor files (for D4).

Input: Lines
    Lines : list of string, containing the all lines in the ASCII file
        Each line is an lement in the string

Output: lineR, AS[0], AS[1], FS[0], FS[1], FS[2], SS[0]
    lineR: index of the last line with R's
    AS: list with the indexes of lines with A's
    FS: list with the indexes of lines with F's
    SS: list with the indexes of lines with S's

Author: Gabriel Cuello
Date:   Jan 2022
--------------------------------------------------------------------------------
    """
    a = 0
    f = 0
    s = 0
    FS = []
    AS = []
    SS = []
    for i in range(len(Lines)):
        if Lines[i][0:10] == "RRRRRRRRRR":
            lineR = i
        if Lines[i][0:10] == "FFFFFFFFFF":
            FS.append(i)
            f += 1
        if Lines[i][0:10] == "AAAAAAAAAA":
            AS.append(i)
            a += 1
        if Lines[i][0:10] == "SSSSSSSSSS":
            SS.append(i)
            s += 1
    return lineR, AS[0], AS[1], FS[0], FS[1], FS[2], SS[0]
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def readD4numorASCII(filename):
    """
Read one numor in ASCII format.

Create a dictionary with the information in the file header and a 2D-matrix
with the registered counts.

Input: filename
    filename: string, the filename of the numor to be read.

Output: data, counts
    data: A dictionary with the information in the file header.
    counts: A 2D-matrix (ndet x ncell) containing the counts of detection cells.

Notes:
    (1) This function is already adapted for the electronically-doubled number
        of cells, i.e., 128 cell instead of 64 for each detection bank.
    (2) Inside the function there are 7 variables containing the linenumbers
        where the lines with R, A, F and S are found in the numor file.
    (3) The variable cdel contains the number of columns between two
        consecutive numbers in a line of floats.
    (4) In the case of future format modifications of the ASCII numors, these
        variables should be updated. From 2023 the ASCII format for D4 data will
        be discontinued, so no changes to this format are expected.
    (5) The labels of the variables given in the numor file are used as keys for
        the dictionary containing the header information.

Author: Gabriel Cuello (ILL)
Date:   Jan 2022
--------------------------------------------------------------------------------
    """
#--------1---------2---------3---------4---------5---------6---------7---------8
# Reading the file and creating the list Lines, for which each element is a
# line in the file, stored as a string.
    with open(filename, "r") as rawdata:
        Lines = rawdata.readlines()

# Automatically determines the reference lines in a numor
    lineR, lineAL, lineAI, lineFD, lineFS, lineFM, lineSI = getRefLines(Lines)
# number of columns between 2 consecutive numbers in a line of floats
    cdel = 16

# Creating a dictionary which will contain the information about the numor.
# The keys are defined depending on the information contained in that line.
# This definition is done in an arbitrary but natural way, here in the code.
# The keys names could be modified if necessary.
    data = {}
    col = (Lines[lineR + 1].strip(" ")[:-1]).split()
    data["numor"] = str(col[0]).zfill(6)
    data["label"] = Lines[lineAL + 2][:-1]
    data["user"] = Lines[lineAI + 2].strip(" ")[20:-1]
    data["LC"] = Lines[lineAI + 3].strip(" ")[20:-1]
    data["title"] = Lines[lineAI + 4][20:-1]
    data["subtitle"] = Lines[lineAI + 5][20:-1]
    data["proposal"] = Lines[lineAI + 6][20:-1]
    data["startTime"] = Lines[lineAI + 7][20:38]
    data["endTime"] = Lines[lineAI + 7][60:79]

# Here starts the reading of the floating numbers blocks.
# Each block starts with a line of 80 'F'.
# The keys for the dictionary are automatically assigned with the lables found
# in the numor file. Those labels have been agreed between SCI and IR, and
# cannot be changed. To change them a new agreement must be found with SCI.
# Only the SCI team can change labels in a numor file. This is true in
# general for any change in the format of numors, either ASCII or Nexus.

# DETECTOR: Reading the detector block from the numor file.
# This block contains a information about the instrument configuration.
# Some of the most useful entries are: monitor couts, counting time, 2theta,
# counts on each detection bank, total number of counts, etc.
    lref = (lineFD + 3)  # Reference line: the line where the 1st label appears.
    col = (Lines[lineFD + 1].strip(" ")[:-1]).split()
    ldel = int(col[1]) - 1  # Nbr of lines between the labels and data blocks.
    for i in range(ldel):
        for j in range(5):
            parameter = None
            while parameter is None:
                try:
                    parameter = float(
                        (Lines[lref + ldel + i][1 + j * cdel : (j + 1) * cdel]
                        ).lstrip(" "))
                except Exception:
                    parameter = float(0.0)
            data[(Lines[lref + i][1 + j * cdel : (j + 1) * cdel]).lstrip(" ")
                ] = parameter

# SAMPLE ENVIRONNEMENT AND REACTOR PARAMETERS
# This block contains information about the sample environment (temperatures,
# pressures, etc.).
# There is also information about the reactor power and reactor cycle.
# In fact, the reactor cycle is always 0 (at least at present, January 2022).
    lref = lineFS + 3  # Reference line: the line where the 1st label appears.
    col = (Lines[lineFS + 1].strip(" ")[:-1]).split()
    ldel = int(col[1]) - 1  # Nbr of lines between the labels and data blocks.
    for i in range(ldel):
        for j in range(5):
            parameter = None
            while parameter is None:
                try:
                    parameter = float(
                        (Lines[lref + ldel + i][1 + j * cdel : (j + 1) * cdel]
                        ).lstrip(" "))
                except Exception:
                    parameter = float(0.0)
            data[(Lines[lref + i][1 + j * cdel : (j + 1) * cdel]).lstrip(" ")
                ] = parameter

# MONOCHROMATOR
# This block contains all the parameters from the monochromator.
# Many parameters are yet properly updated (Jan 2022), so be careful with this.
# An useful parameter is the incident energy and the d-spacing. These values
# could be used to determine the default working wavelength.
    lref = lineFM + 3  # Reference line: the line where the 1st label appears.
    col = (Lines[lineFM + 1].strip(" ")[:-1]).split()
    ldel = int(col[1]) - 1 # Nbr of lines between the labels and data blocks.
    for i in range(ldel):
        for j in range(5):
            parameter = None
            while parameter is None:
                try:
                    parameter = float(
                        (Lines[lref + ldel + i][1 + j * cdel : (j + 1) * cdel]
                        ).lstrip(" "))
                except Exception:
                    parameter = float(0.0)
            data[(Lines[lref + i][1 + j * cdel : (j + 1) * cdel]).lstrip(" ")
                ] = parameter

# COUNTS
# These are 9 consecutive blocks containing the counts for each detector bank.
# Each block starts with a line of 80 'S'.
# The next line contains the current bank, the remaining banks, the total
# number number of banks (9) and the numor.
# Then, there is a line with 80 'I', preceding the block of integer numbers.
# The next line contains a single integer: number of cells in a detector (64).
# Then, the individual counts for each cell are written (10 values per line).
# Note that here the script determenines the number of detectors (ndet) and
# the number of cells per dtector (ncell).
# At presesent (Jan 2022), there are 9 detectors and 64 cells per detector.
# This could be changed in a near future to 9 x 128. Thanks to the fact of
# reading this information frmo the numor, this change should be transparent
# for this script, provided that we keep the same format for recording the
# counts, i.e., a single line with the number of cells per detctor (128) and
# 10 values per line with the counts.
    lref = lineSI  # Reference line: the first line with 80 'S'
    col = (Lines[lineSI + 1].strip(" ")[:-1]).split()
    ndet = int(col[2])  # number of detection banks.
    data["ndet"] = ndet
    ncell = int(Lines[lref + 3].strip(" ")[:-1])
    data["ncell"] = ncell

# Initialising the matrix for the counts: 9x65 (or 9x129)
    counts = np.zeros((ndet, ncell))

# A first loop over the 9 detectors
    for det in range(ndet):
    # There are 10 numbers on each line.
    # Then this loop goes up to 6 (or 12), reading 10 numbers at each loop.
        for j in range(int(ncell / 10)):
            # col is a list of 10 elements containing the numbers
            col = (Lines[lref + 4 + j + 11 * det].strip(" ")[:-1]).split()
            # The numbers are assigned to the matrix counts
            for i in range(len(col)):
                counts[det, i + 10 * j] = col[i]
    # Now, the remaining line is read and data stored in the list col.
    # This line contains 4(8) numbers, depending on the nbr of cells, 64(128).
        col = (
            Lines[lref + 4 + int(ncell / 10) + 11 * det].strip(" ")[:-1]
        ).split()
        # The numbers are assigned to the matrix counts
        for i in range(len(col)):
            counts[det, i + 10 * int(ncell / 10)] = col[i]

    print("Numor {} loaded without errors.".format(data["numor"]))
    print(2 * " ", data["subtitle"])
    print("  Starts: {}  Ends: {}".format(data["startTime"], data["endTime"]))
    print("  Counting time ={:10.6g} s".format(data["CntTime (sec)"]))
    print("  Angle (1st cell 1st det) ={:10.6g} degrees".format(
            data["Theta_lue(deg)"]))
    printRates("monitor",
               data["MonitorCnts"],data["CntTime (sec)"],data["RtrPower (MW)"])
    printRates("detector",
        data["TotalCnts"],data["CntTime (sec)"],data["RtrPower (MW)"])
    print()

    return data, counts
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def load_nxs(nxs_path, ncell=64):
    """
Read one numor in Nexus format.

Creates a dictionary with the information in the metadata and a 2D-matrix with
the registered counts.

Equivalent to readD4numorASCII() but for nexus files.

Input: nxs_path, ncell
    nxs_path: string
        The filename and path of the nexus numor to be read (ending with ".nxs")
    ncell: int
        The number of detection cells of the instrument. To be modified to 128
        only when needed.

Output: metadata, counts
    metadata: A dictionary with the information in the file metadata.
    counts: A 2D-matrix (ndet x ncell) containing the counts of detection cells.

Notes:
    (1) This function is adapted for the electronically-doubled number of cells,
        i.e., 128 cell instead of 64 for each detection bank.
        In case of using 128 cells, modify ncell to 128.
    (2) This function uses h5py to read the nexus files. The nexus files still
        contains more information than the one loaded here. This function only
        load the information necessary to run all functions in this module.
    (3) The labels of the variables given in the numor file are used as keys for
        the dictionary containing the header information.

Author: José Robledo (CAB-CNEA)
Date:   Jan 2022
--------------------------------------------------------------------------------
    """
    # read from nexus
    with h5py.File(nxs_path, "r") as nxs:
        entry0 = nxs["entry0"]

        # create metadata dictionary
        metadata = {
            "label": f"d4 {entry0['user/name'][0].decode('utf-8')}{entry0['user/namelocalcontact'][0].decode('utf-8')}{entry0['start_time'][0].decode('utf-8')}",
            "user": entry0["user/name"][0].decode("utf-8"),
            "LC": entry0["user/namelocalcontact"][0].decode("utf-8"),
            "proposal": entry0["user/proposal"][0].decode("utf-8"),
            "ndet": 9,
            "ncell": ncell,
            "RtrPower (MW)": 1,
            "numor": str(entry0["run_number"][0]),
            "subtitle": entry0["experiment_identifier"][0].decode("utf-8"),
            "title": entry0["title"][0].decode("utf-8"),
            "startTime": entry0["start_time"][0].decode("utf-8"),
            "endTime": entry0["end_time"][0].decode("utf-8"),
            "MonitorCnts": entry0["monitor/data"][0, 0, 0],
            "CntTime (sec)": entry0["time"][0],
            "TotalSteps": entry0["data_scan/total_steps"][0],
            "Theta_lue(deg)": entry0["instrument/2theta/value"][0],
            "Theta_des(deg)": entry0["instrument/2theta/target_value"][0],
            "Omega(deg)": entry0["instrument/omega/value"][0],
            "A1": entry0["instrument/A1/value"][0]
        }

        for name, data in zip(
            entry0["data_scan/scanned_variables/variables_names/property"],
            entry0["data_scan/scanned_variables/data"]
        ):
            if name.decode("utf-8") == "TotalCount":
                name = "TotalCnts"
            else:
                name = name.decode("utf-8")
            metadata[name] = data[0]

        # create counts
        counts = (
            entry0["data_scan/detector_data/data"][0]
            .squeeze()
            .reshape(metadata["ndet"], metadata["ncell"])
        )
    return metadata, counts
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def readParam(parametersFile):
    runInfo = {}

#--------1---------2---------3---------4---------5---------6---------7---------8
# This block rarely changes
#   Angular range covered by a detection bank. At present this value is 8 deg.
    runInfo["angular_range_bank"] = 8.0
# Dead-time (in s) for monitor and detector.
#   If these values are equal to 0, no dead-time correction is performed.
    runInfo["dead_time"] = (2.4e-6, 7.0e-6)
# Tolerance (in cell units) for warning/error in angle positioning
#   The requested angle is compared to the read angle.
#   If the difference is bigger than first value, only a warning and continue.
#   If this difference is bigger than second value, error message and stop.
    runInfo["cellTolerance"] = (0.1, 1.0)
    # Format of the numors: can be 'ASCII' or 'Nexus'
    runInfo["dataFormat"] = "Nexus"
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
# This block can change for each experiment.
# Path (relative to the working directory) to the rawdata
    runInfo["path_raw"] = "rawdata/"
# Extension for the output data files:
#   .reg:  Files in D4 format (angle, counts, error, Q)
#   .adat: Files in CORRECT format (angle, counts, error)
#   .qdat: Files in CORRECT format (Q, counts, error)
#   .cdat: Files with single numor (angle, counts, error, Q, cell)
#   .nxs:  Files in Nexus format, typically the rawdata
#   .log:  Logfiles
    runInfo["ext"] = (".reg", ".adat", ".qdat", ".cdat", ".nxs", ".log")
# Efficiency file, containing the relative efficiencies (I=counts/eff).
#   'ones': replace all efficiencies by 1.
    runInfo["efffile"] = "effd4c.eff"
# Shift file, containing the small angular correction for each detection bank.
#   'zeros': replace all shifts by 0.
#   'manual': shifts given in function getDec (in module readD4.py).
    runInfo["decfile"] = "dec.dec"
    runInfo['logfile'] = "d4creg"+runInfo["ext"][5]
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
# This block can change for every experiment, and even for some samples.
# Defines the normalisation mode and the values used to normalise.
#   0: 'monitor' or 'time'
#   1: normalising time in seconds. Normal reactor power, 80s approx 1E6 mon.
#   2: normalising monitor counts. Usually 1E6 monitor counts.
    runInfo["normalisation"] = ("monitor", 80, 1000000)
# Write individual files for each numor depending on 'writeNumor' value (0/1).
#   0: no individual numor files are created.
#   1: an individual numor file is created for each numor in the range.
#   If only one numor in the range, 'writeNumor' is forced to 1.
    runInfo["writeNumor"] = 0
    runInfo["plotDiff"] = 0
# Zero-angle correction (deg) and working wavelength (A).
    runInfo["twotheta0"] = -0.1013
    runInfo["wavelength"] = 0.49857
# Binning in angular and Q scales.
    runInfo["angular_scale"] = (0.0, 140.0, 0.125)  # Angular scale (degrees)
    runInfo["q_scale"] = (0.0, 23.8, 0.02)  # Q-scale (1/A)
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
# This block changes for every sample
# Name identifying the sample. Root for the output filenames (root.ext).
    runInfo["file"] = "vanadium"
# Numors to be included. Individual numors or range of numors are accepted.
# Examples:
#   - ['387229-387288']
#   - ['387229','387231','387231','387235-387270','387275','387280-387288']
#   - ['387229']
# runInfo['numorLst'] = ['387229-387230']
    runInfo["numorLst"] = ["387229-387288"]


    with open("../logfiles/d4creg.log", "w") as file:
        file.write("# d4creg.log\n\n")

    with open(parametersFile, "r") as parfile:
        lines = parfile.readlines()

    for i in range(len(lines)):
        if (len(lines[i]) > 1):
            first = (lines[i][0] != "#") and (lines[i][0] != "!") and (lines[i][0] != "<")
            if first:
    #            print (first)
                print ("Wrong input in line: ",i+1," file: ",parametersFile)
                sys.exit()


    for i in range(len(lines)):
    #    print (lines[i][0:-1])
        if (lines[i][0] == "#") or (lines[i][0] == "!"):
    #        print ("Comment line",len(lines[i]))
            pass
        elif len(lines[i]) == 1:
    #        print("Blank line")
            pass
        elif (lines[i][0] == "<") and (lines[i][4] == ">"):
            line = lines[i].split(" ")
            if line[0] == "<tol>":
                runInfo["cellTolerance"] = (float(line[1]),float(line[2]))
                print ("Cell tolerance: {} cells for warning, {} cells for error"
                       .format(*runInfo["cellTolerance"]))
            if line[0] == "<tau>":
                runInfo["dead_time"] = (float(line[1]),float(line[2]))
                print ("Dead times: {} s for monitor, {} s for detector"
                       .format(*runInfo["dead_time"]))
            if line[0] == "<ext>":
    #            runInfo["ext"] = (line[j] for j in range(1,7))
                runInfo["ext"] = (line[1],line[2],line[3],line[4],line[5],line[6])
                print ("Extensions: {}, {}, {}, {}, {}, {}"
                       .format(*runInfo["ext"]))
            if line[0] == "<eff>":
                runInfo["efffile"] = line[1]
                print ("Efficiency file: {}".format(runInfo["efffile"]))
            if line[0] == "<dec>":
                runInfo["decfile"] = line[1]
                print ("Shifts file: {}".format(runInfo["decfile"]))
            if line[0] == "<log>":
                runInfo["logfile"] = line[1]
                print ("Shifts file: {}".format(runInfo["decfile"]))
            if line[0] == "<fmt>":
                runInfo["dataFormat"] = line[1]
                print ("Data format: {}".format(runInfo["dataFormat"]))
            if line[0] == "<asc>":
                runInfo["angular_scale"] = (float(line[1]),float(line[2]),float(line[3]))
                print ("Angular scale: from {} deg to {} deg in steps of {} deg"
                       .format(*runInfo["angular_scale"]))
            if line[0] == "<qsc>":
                runInfo["q_scale"] = (float(line[1]),float(line[2]),float(line[3]))
                print ("Q scale: from {} 1/A to {} 1/A in steps of {} 1/A"
                       .format(*runInfo["q_scale"]))
            if line[0] == "<wri>":
                if line[1] == "True":
                    runInfo["writeNumor"] = 1
                else:
                    runInfo["writeNumor"] = 0
                print ("Write individual files for each numor: {} {}"
                       .format(line[1],runInfo["writeNumor"]))
            if line[0] == "<plo>":
                runInfo["plotDiff"] = 0
                if line[1] == "True":
                    runInfo["plotDiff"] = 1
                else:
                    runInfo["plotDiff"] = 0
                print ("Plot diffractograms: {} {}"
                       .format(line[1],runInfo["plotDiff"]))
            if line[0] == "<wle>":
                runInfo["wavelength"] = float(line[1])
                print ("Incident wavelength: {} A"
                       .format(runInfo["wavelength"]))
            if line[0] == "<zac>":
                runInfo["twotheta0"] = float(line[1])
                print ("Zero-angle: {} deg"
                       .format(runInfo["twotheta0"]))
            if line[0] == "<rdp>":
                runInfo["path_raw"] = line[1]
                print ("Relative path for rawdata: {}".format(runInfo["path_raw"]))
            if line[0] == "<nor>":
                runInfo["normalisation"] = (line[1],float(line[2]),float(line[3]))
                print ("Normalisation mode: {}".format(runInfo["normalisation"][0]))
                print ("Normalisation constants: {1} s or {2} monitor counts"
                       .format(*runInfo["normalisation"]))
            if line[0] == "<out>":
                runInfo["file"] = line[1]
                print ("Base name for output files: {}".format(runInfo["file"]))

            if line[0] == "<num>":
                runInfo["numorLst"] = [line[1]]
                print ("List of numors: {}".format(runInfo["numorLst"]))

            if line[0] == "<add>":
                runInfo["numorLst"].append(line[1])
                print ("List of numors: {}".format(runInfo["numorLst"]))

            elif line[0] == "<run>":
                print ("Calling d4creg...")
                d4creg(runInfo)

        else:
            print ("Input error in line: ",i+1," file: ",parametersFile)
            sys.exit()
    return runInfo
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def getEff(filename="effd4c.eff", ndet=9, ncells=64):
    """
Generate the relative efficiencies.

Reads the relative efficiencies from the given file (default 'effd4c.eff').
The reserved name 'ones' fixes all efficiencies to 1.

Input: filename, ndet, ncells
    filename : string, optional
        The filename for the relative efficiencies. Default: 'effd4c.eff'.
        If the string is equal to 'ones': all the efficiencies are 1.
    ndet : integer, optional
        Number of detector banks. The default is 9.
    ncells : integer, optional
        Number of cells in one detector bank. The default is 64.

Output:
    efficiencies: a 2D-matrix with the efficiencies
                  axis 0: detector bank
                  axis 1: cell

Note:
    (1) The efficiency is the coefficient that multiplies the expected counting
        value to give the measured value, i.e., I_expected = I_meas/eff.

Author: Gabriel Cuello
Date:   Jan 2022
--------------------------------------------------------------------------------
    """
    # Initialising the matrix for the efficiencies: 9x64 (or 9x128)
    efficiencies = np.ones((ndet, ncells)) # Initialise the vector with 1's
    if filename == "ones":
        print("No relative efficiencies")
    else:
        with open(filename, "r") as shifts:
            Lines = shifts.readlines()
        for i in range(len(Lines)):
            if "#" not in Lines[i]:  # Only the lines without # are treated.
                row = (Lines[i].strip(" "))[0:-1]
                if len(row) > 0:  # Only the no-empty lines are treated
                    col = row.split()  # This method split the line
                    if float(col[2]) <= 0:
                        col[2] = "nan"
                    efficiencies[int(col[0]) - 1, int(col[1]) - 1] = col[2]
    return efficiencies
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def getDec(filename="dec.dec", ndet=9):
    """
Reads from a file the angular shifts for each detection bank.

The shifts can be manually introduced within this function using the reserved
name 'manual'. The reserved name 'zeros' fixes all shifts to 0.

Input: filename, ndet
  - filename : string, optional
        The filename for the angular shifts. The default is 'dec.dec'.
        If the string is:
            - 'zeros': all the shifts are zero.
            - 'manual': shifts are fixed to the values inside this function.
  - ndet : integer, optional
        Number of detector banks. The default is 9.

Output: zero
    zero : array with the 9 shift values (in deg).

Author: Gabriel Cuello
Date:   Jan 2022
--------------------------------------------------------------------------------
    """
    if filename == "zeros":
        print("No angular shifts")
        zero = np.zeros(ndet) # This initialise the vector with zeros
    elif filename == "manual":
        print("Built-in shifts:")
        zero = np.array([ 0.000,-0.027,-0.026,
                         -0.087,-0.125,-0.160,
                         -0.220,-0.250,-0.340])
    else:
        zero = np.zeros(ndet)
        with open(filename, "r") as shifts:
            Lines = shifts.readlines()
        for i in range(len(Lines)):
            if "#" not in Lines[i]:  # Only the lines without # are treated.
                row = (Lines[i].strip(" "))[0:-1]
                if len(row) > 0:  # Only the no-empty lines are treated
                    columns = row.split()  # This method split the line
                    zero[int(columns[0]) - 1] = float(columns[1])
    return zero
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def getErrorsNumor(counts):
    """
Calculates the errors corresponding to the detector counts.

The errors are calculated assuming a Poisson statistics, i.e., as the square
root of the number of counts.

Input: counts
  - counts : 2D-matrix of floats (ndet x ncell)
        Contains the registered counts for each cell of each detector

    Returns
    -------
  - errors : 2D-matrix of floats (ndet x ncell)
        Contain the square root of the counts.

Author: Gabriel Cuello
Date:   Jan 2022
--------------------------------------------------------------------------------
    """
    # Initialising the matrix for the errors. Same dimensions as counts matrix.
    errors = np.zeros((counts.shape[0], counts.shape[1]))
    errors = np.sqrt(counts)
    return errors
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def getNormalFactors(monitor, time):
    """
Defines the normalisation factors.

Definition of a 2x2 matrix containing the normalisation factors i.e., monitor
counts or counting time
    - 1st row: monitor counts, error of monitor counts
    - 2nd row: counting time, error in counting time (in seconds)

The row index is the normalisation method. Thus,
normal[norm][0], normal[norm][1] = value, error
where norm = 0 --> normalisation by monitor
      norm = 1 --> normalisation by counting time

Input: monitor, time
    monitor : float
        monitor counts
    time : float
        counting time (in seconds)

Output: normal
  - normal: normalisation factors and errors (2x2 matrix)

Notes:
  - The error in counting time is supposed to be 0.01 seconds.

Author: Gabriel Cuello
Date:   Jan 2022
--------------------------------------------------------------------------------
    """
    normal = np.zeros((2, 2))
    normal[0][0] = monitor  # Monitor counts
    normal[0][1] = np.sqrt(monitor)  # Error monitor counts
    normal[1][0] = time  # Counting time
    normal[1][1] = 0.01  # Error counting time (s)
    return normal
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def normalise(counts,errors,normal,
              norm_type="monitor",norm_time=120,norm_mon=1000000):
    """
Normalise the detector counts by monitor counts or counting time.

Normalises the counts and errors by monitor or counting time.
    - monitor:  counts/monitor * norm_mon
    - time:  counts/time * norm_time

Input: counts,errors,normal,norm_type=,norm_time,norm_mon
  - counts : 2D matrix (ndet x ncell)
        Matrix containing the counts of a single numor
  - errors : 2D matrix (ndet x ncell)
        Matrix containing the errors of a single numor
  - normal : 2D matrix (2 x 2)
        Matrix containing the normalisation factors and errors
  - norm_type : string, optional
        Defines the normalusaion mode. The default is 'monitor'.
        'monitor': normalisation by monitor
        'time': normalisation by counting time
  - norm_time : float, optional
        Counting time for normalisation (in seconds). The default is 120 s.
  - norm_mon : float, optional
        Monitor counts for normalisation. The default is 1000000.

Output:
  - counts : 2D matrix (ndet x ncell)
        Matrix containing the normalised counts of a single numor
  - errors : 2D matrix (ndet x ncell)
        Matrix containing the normalised errors of a single numor

Author: Gabriel Cuello
Date:   Jan 2022
--------------------------------------------------------------------------------
    """
    norm = [norm_mon, norm_time]
    ntype = 0  # normalisation by monitor
    if norm_type == "time":
        ntype = 1  # normalisation by time
    counts = norm[ntype] * counts / normal[ntype][0]
    relative_error = np.sqrt(
        (errors / counts) ** 2 + (normal[ntype][1] / normal[ntype][0]) ** 2
    )
    errors = counts * relative_error
    return counts, errors
#--------1---------2---------3---------4---------5---------6---------7---------8


#--------1---------2---------3---------4---------5---------6---------7---------8
def getAngles(counts, ttheta_ref, zeros, cell_step=0.125, det_step=15.0):
    """
Determine the angles and cell labels fro a gove numor.

For a given numor, i.e., a reference scattering angle, this function
returns the list of angles for each cell of the 9 detection blocks.
An identification of each cell is produced too, as:
    det * 1000 + cell
in this way 7050 is the 50th cell of detector 7.

Input:
  - counts : 2D matrix (ndet x ncell)
        Counts for a given numor.
        Used only to determine the total number of cells.
  - ttheta_ref : float
        Reference angle for the 1st cell of the 1st detector (in deg)
  - zeros : array of floats
        The angular shifts of each detection block (in deg)
  - cell_step : float, optional
        The angular range covered by 1 cell, in degrees. Default: 0.125 deg.
  - det_step : float, optional
        Angular distance between the 1st cells of 2 detection blocks, in deg.
        The default is 15.0 degrees.

Output: angles, cells
  - angles : array of angles, float
        Angle of each cell
  - cells : array of identifiers, float
        Indentifier for each cell

Author: Gabriel Cuello
Date:   Jan 2022
--------------------------------------------------------------------------------
    """
    angles = np.zeros((counts.shape[0], counts.shape[1]))
    cells = np.zeros((counts.shape[0], counts.shape[1]))
    ndet = counts.shape[0]
    ncel = counts.shape[1]
    for det in range(ndet):
        for cel in range(ncel):
            angles[det][cel] = (
                ttheta_ref + zeros[det] + (det * det_step + cel * cell_step)
            )
            cells[det][cel] = (det + 1) * 1000 + cel + 1
    return angles, cells
#--------1---------2---------3---------4---------5---------6---------7---------8


#--------1---------2---------3---------4---------5---------6---------7---------8
def saveOneNumor(x, y, e, q, c, head, runInfo):
    """
Save an individual file for a single numor.

Create a file with the numor name and extension .cdat.
The file contains a header with basic inofrmation, then five columns with
angle (deg), counts, error, Q (1/Å), and cell.

Input: x, y, e, q, c, head, runInfo
  - x : float
        Angle in degrees
  - y : float
        Counts
  - e : float
        Errors
  - q : float
        Momentum transfer (1/A)
  - c : integer
        Cell identifier
  - head : dictionary
        Contains the basic information about the numor.
  - runInfo : dictionaty
        Contains the basic information about the running parameters.

Output: None
    A file with extemsion .cdat is created.

Author: Gabriel Cuello
Date:   Jan 2022
--------------------------------------------------------------------------------
    """
    filename = head["numor"] + runInfo["ext"][3]
    with open(filename, "w") as datafile:
        datafile.write("# {} \n".format(filename))
        datafile.write("# Sample: {}\n".format(head["subtitle"]))
        datafile.write("# Starts: {}  Ends: {} \n".format(
                head["startTime"], head["endTime"]))
        datafile.write(
            "# Counting time: {:8.4f} s\n".format(head["CntTime (sec)"]))
        datafile.write("# Monitor: {:10.2f} counts ({:8.2f} c/s)\n".format(
                head["MonitorCnts"],head["MonitorCnts"] / head["CntTime (sec)"])
                      )
        datafile.write("# Detector: {:10.2f} counts ({:8.2f} c/s)\n".format(
                head["TotalCnts"], head["TotalCnts"] / head["CntTime (sec)"]))
        datafile.write("#" + 80 * "-" + "\n")
        datafile.write("# Angle(deg)"+ 6 * " "+ "Counts"+ 14 * " "
            + "Error"+ 8 * " "+ "Q(1/Å)"+ 6 * " "+ "Cell \n")
        for i in range(len(x)):
            datafile.write(
                "{: 9.3f} {:18.6f} {:18.6f} {:9.3f} {:9.0f}".format(
                    x[i], y[i], e[i], q[i], c[i])+ "\n")
        print("File {} saved.".format(filename))

    return
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def getOneNumor(numorfile, runInfo):
    """
Reads a numor file and create the counting lists.

This function puts together all the information coming from a single numor.
But also:
    - Check the angular positioning, comparing the requested angle to the
      read one. If the difference is greater than a given number of cells
      (1 cell), produces an error message and stops the program.
      A second threshold (0.1 cell) produces only a warning on the screen.
      These two limits are controlled via runInfo dictionary.
    - Make the deadtime correcton on monitor and detector counts.
    - Divide the registered counts by the relative efficiency of cells.
    - Correct the angles by the angular shifts of each detection bank.

Required functions:
    readD4numorASCII: Read one numor in ASCII format
    getDTC_mcounts:   Correct monitor counts by deat-time
    getDTC_dcounts:   Correct detector counts by deat-time
    printRates:       Print the counting rates on the screen
    getEff:           Read the relative efficiencies
    getErrorsNumor:   Calculate the errors in detector counts
    getNormalFactors: Determine the normalisation factors
    normalise:        Perform the data normalisation
    getDec:           Read the angular shift of each detection bank
    getAngles:        Determine the angles and cell ID for each cell
    ang2q:            Convert scattering angles in Q
    saveOneNumor:     Save on disk a single numor

Input: numorfile, runInfo
  - numorfile : string
        The name identifying the numor.
  - runInfo : dictionary
        Contains basic information about the running parameters.

Output: x, y, e, q, c, head
  - x : float
        Angle in degrees
  - y : float
        Counts
  - e : float
        Errors
  - q : float
        Momentum transfer (1/A)
  - c : integer
        Cell identifier
  - head : dictionary
        Contains the basic information about the numor.

Author: Gabriel Cuello
Date:   Jan 2022
--------------------------------------------------------------------------------
    """
    write = runInfo["writeNumor"]      # write individual numors: 0 or 1
    efffile = runInfo["efffile"]       # efficiency file
    decfile = runInfo["decfile"]       # shifting file
    norm_type = runInfo["normalisation"][0]  # 'monitor' or 'time'
    norm_time = runInfo["normalisation"][1]  # time for normalisation
    norm_mon = runInfo["normalisation"][2]   # monitor counts for normalisation
    dtm = runInfo["dead_time"][0]      # monitor dead-time
    dtd = runInfo["dead_time"][1]      # detector dead-time
    cell_tol = runInfo["cellTolerance"]      # cell tolerance: warning/error

#    print(80 * "-")
    if runInfo["dataFormat"] == "Nexus":
        # Reading the numor in Nexus format
        head, counts = load_nxs(numorfile + runInfo['ext'][4])
    else:
        # Reading the numor in ASCII format
        head, counts = readD4numorASCII(numorfile)

    # Checking the angular positioning
    # For 64 cells, 0.125 deg
    angular_range_cell = runInfo["angular_range_bank"] / head["ncell"]
    deltaTheta = np.abs(head["Theta_lue(deg)"] - head["Theta_des(deg)"])
    if deltaTheta > cell_tol[1] * angular_range_cell:
        with open('../logfiles/d4creg.log', 'a') as file:
            file.write("   ERROR! The angle differs from the requested angle by\n")
            file.write("          more than {} cell\n".format(cell_tol[1]))
            file.write("          Angle = {} deg, requested angle = {} deg\n".format(
                head["Theta_lue(deg)"], head["Theta_des(deg)"]))
        print("   ERROR! The angle differs from the requested angle by")
        print("          more than {} cell".format(cell_tol[1]))
        print("          Angle = {} deg, requested angle = {} deg".format(
                head["Theta_lue(deg)"], head["Theta_des(deg)"]))
        print(0/0)  # This terminates the program
    elif deltaTheta > cell_tol[0] * angular_range_cell:
        with open('../logfiles/d4creg.log', 'a') as file:
            file.write("   WARNING! The angle differs from the requested angle by\n")
            file.write("            more than {} cell\n".format(cell_tol[0]))
            file.write("            Angle = {} deg, requested angle = {} deg\n".format(
                head["Theta_lue(deg)"], head["Theta_des(deg)"]))
        print("   WARNING! The angle differs from the requested angle by ")
        print("            more than {} cell".format(cell_tol[0]))
        print("            Angle = {} deg, requested angle = {} deg".format(
                head["Theta_lue(deg)"], head["Theta_des(deg)"]))

    # Dead-time corrections
    with open('../logfiles/d4creg.log', 'a') as file:
        file.write("Dead-time corrections\n")
#    print("Dead-time corrections")
    head["MonitorCnts"] = getDTC_mcounts(
        head["MonitorCnts"], head["CntTime (sec)"], dead_time=dtm)
    counts, head["TotalCnts"] = getDTC_dcounts(
        counts, head["TotalCnts"], head["CntTime (sec)"], dead_time=dtd)

    printRates("monitor",
        head["MonitorCnts"],head["CntTime (sec)"],head["RtrPower (MW)"])
    printRates("detector",
        head["TotalCnts"],head["CntTime (sec)"],head["RtrPower (MW)"])

    # Relative efficiencies correction
    efficiency = getEff(filename=efffile,
                        ndet=head["ndet"], ncells=head["ncell"])
    counts = counts / efficiency  # raw data are divided by the efficiencies
    # Calculation of the experimental errors
    errors = getErrorsNumor(counts)
    # Normalising the data
    normal = getNormalFactors(head["MonitorCnts"], head["CntTime (sec)"])
    counts, errors = normalise(counts,errors,normal,
        norm_type=norm_type,norm_time=norm_time,norm_mon=norm_mon)
    # Calculating the angular coordinates
    zeros = getDec(filename=decfile, ndet=head["ndet"])
    #   2theta = 2theta.raw + zeros[det]
    angles, cells = getAngles(counts, head["Theta_lue(deg)"], zeros)

    with open('../logfiles/d4creg.log', 'a') as file:
        file.write(80 * "-" + "\n")
#    print(80 * "-")

    x, y, e, c = [], [], [], []
    for det in range(head["ndet"]):
        for cell in range(len(counts[0, :])):
            x.append(angles[det][cell])
            y.append(counts[det][cell])
            e.append(errors[det][cell])
            c.append(cells[det][cell])

    ang = np.array(x)
    q = ang2q(ang, wlength=runInfo["wavelength"])
    if write == 1:
        saveOneNumor(x, y, e, q, c, head, runInfo)

    return x, y, e, q, c, head
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def getNumorFiles(numorLst):
    """
Creates a detailed list of numors.

Take a short syntax list of numors and returns a detailed list of numors
(strings that are used as filenames for readung numors).

Input: numorList
  - numorLst : List of numors in short syntax (strings)
        Each element can be a single numor (6 characters) or a range of numors
        in the format 111111-222222, where 111111 and 222222 are the first and
        last numors in the range. The last numor must be bigger or equal to the
        first numor.

Output: numorFiles
  - numorFiles : Detailed list of numors

author: Gabriel Cuello
date:   Jan 2022
--------------------------------------------------------------------------------
    """
    numorFiles = []
    lenNumor = 6  # Default length of a numor name, i.e., 6 characters
    for i in range(len(numorLst)):
        if len(numorLst[i]) == lenNumor:
            numorFiles.append(numorLst[i])
        elif (len(numorLst[i]) == 2 * lenNumor + 1):  # this corresponde to a range
            firstNum = int(numorLst[i][0:lenNumor])
            lastNum = int(numorLst[i][lenNumor + 1 : 2 * lenNumor + 1])
            if firstNum > lastNum:
                with open('../logfiles/d4creg.log', 'a') as file:
                    file.write("ERROR with the range \n", numorLst[i])
                    file.write("      The last numor must be less than or equal\n")
                    file.write("      to the first one.\n")
                print("ERROR with the range ", numorLst[i])
                print("      The last numor must be less than or equal")
                print("      to the first one.")
            for j in range(firstNum, lastNum + 1):
                numorFiles.append(str(j).zfill(lenNumor))
        else:
            with open('../logfiles/d4creg.log', 'a') as file:
                file.write("ERROR in the list of numors (getNumorFiles function)\n")
                file.write("      It is likely a problem with a numor not having six\n")
                file.write("      characters.\n")
            print("ERROR in the list of numors (getNumorFiles function)")
            print("      It is likely a problem with a numor not having six")
            print("      characters.")
#     with open('../logfiles/d4creg.log', 'a') as file:
#         file.write("List of numors included in the diffractogram ({} in total):\n".format(
#             len(numorFiles)))
#         file.write(numorFiles)
#         file.write("\n")

#    print(
#        "List of numors included in the diffractogram ({} in total):".format(
#            len(numorFiles)))
#    print(numorFiles)
#    print()
    return numorFiles
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def getNumors(runInfo):
    """
Puts all numors together in the same matrix.

This function creates a 3D matrix with all the experimental data.

Required functions:
    getNumorFiles: Create the detailed list of numors in the range
    getOneNumor: Read a numor file and create the counting lists

Input: runInfo
    runInfo : dictionary
        Contains basic information about the running parameters.

Output: numor, head
    numor : 3D matrix (nbrNumors x 5 x ndata)
        The 3D matrix numor contains all the data:
        - The first dimension contains the order of the numor, starting from 0
        - The second dimension contains the variable indexes:
              0: the angle (deg)
              1: the counts
              2: the errors
              3: the Q values (1/A)
              4: the cell (20xx corresponds to the cell xx of detector 2)
        - The third dimension contains the values of the corresponding variable
    head : dictionary
        Contains the basic information about the numor.

author: Gabriel Cuello
date:   Jan 2022
--------------------------------------------------------------------------------
    """
#--------1---------2---------3---------4---------5---------6---------7---------8
    numorFiles = getNumorFiles(runInfo["numorLst"])
    if len(numorFiles) == 1:
        runInfo["writeNumor"] = 1

    head = []

    # Reading the first numor
    numorfile = runInfo["path_raw"] + numorFiles[0]
#    print("Numor 1/{}".format(len(numorFiles)),' ',numorFiles[0])

    with open('../logfiles/d4creg.log', 'a') as file:
        file.write(runInfo["file"]+"\n")
        file.write("Numor 1/{}".format(len(numorFiles))+ ' ' + numorFiles[0] + '\n')

    angle, count, error, qval, cell, header = getOneNumor(numorfile, runInfo)
#    print(header["ndet"])
    angle = np.array(angle)
    head.append(header)

    numor = np.zeros((len(numorFiles), 5, len(angle)))

    numor[0, 0, :] = angle - runInfo["twotheta0"]
    numor[0, 1, :] = count
    numor[0, 2, :] = error
    numor[0, 3, :] = qval
    numor[0, 4, :] = cell

    # Reading the other numors, if more than 1 (note that the loop starts at 1)
    for i in range(1, len(numorFiles)):
        numorfile = runInfo["path_raw"] + numorFiles[i]
#        print("Numor {}/{}".format(i + 1, len(numorFiles)),' ',numorFiles[i])

        with open('../logfiles/d4creg.log', 'a') as file:
            file.write("Numor {}/{}".format(i + 1, len(numorFiles))+ ' ' + numorFiles[i] + '\n')

        angle, count, error, qval, cell, header = getOneNumor(
            numorfile, runInfo)
        angle = np.array(angle)
        head.append(header)

        numor[i, 0, :] = angle - runInfo["twotheta0"]
        numor[i, 1, :] = count
        numor[i, 2, :] = error
        numor[i, 3, :] = qval
        numor[i, 4, :] = cell

    return numor, head
#--------1---------2---------3---------4---------5---------6---------7---------8


#--------1---------2---------3---------4---------5---------6---------7---------8
def getDiffA(numor, head, runInfo):
    """
Creates the diffractograms in angular scale.

Produces 10 diffractograms as function of the scattering angle (in deg).
The diffractograms 1 to 9 correspond to the detector of the same index.
The diffractogram 0 corresponds to a single diffractogram regrouping all
detectors.

Required functions:
  - ang2q: Convert scattering angles in Q

Input: numor, head, runInfo
  - numor : 3D matrix (nbrNumors x 5 x nbrAngles)
        The 3D matrix numor contains all the data:
        - The first dimension contains the order of the numor, starting from 0
        - The second dimension contains the variable indexes:
              0: the angle (deg)
              1: the counts
              2: the errors
              3: the Q values (1/A)
              4: the cell (20xx corresponds to the cell xx of detector 2)
        - The third dimension contains the values of the corresponding variable
  - head : dictionary
        Contains the header of each numor
  - runInfo : dictionary
        Contains basic information about the running parameters.

Output: diff
   - diff: a 3D matrix (10 x 5 x NbrAngles)
        - The first dimension is the detector (0 for the total diffractogram)
        - The second dimension contains the variables:
            0: the angle (deg)
            1: the counts
            2: the errors
            3: the weighting factor for each bin
            4: the Q values (1/A)
        - The third dimension contains the values of the corresponding variable

Author: Gabriel Cuello
Date:   Jan 2022
--------------------------------------------------------------------------------
    """
#--------1---------2---------3---------4---------5---------6---------7---------8
    ndet = head[0]["ndet"]    # Number of detectors
    ncell = head[0]["ncell"]  # Number of cells in one detector
    angmin = runInfo["angular_scale"][0]
    angmax = runInfo["angular_scale"][1]
    angstep = runInfo["angular_scale"][2]
    angular_range = runInfo["angular_range_bank"]
    angular_step = angular_range / ncell  # Angular step for 1 cell (in deg)

    if angstep < angular_step:
        print("WARNING: In getDiffA function")
        print("         The angular step should not be smaller than")
        print("         {:6.3f} degrees".format(angular_step))
        print(
            "         Then, it has been changed to {:6.3f} degrees".format(
                angular_step))
        angstep = angular_step

    # Number of angles in the angular binning
    nangles = int((angmax - angmin) / angstep)

    # Defining the matrix that will contain the 10 diffractograms
    diff = np.zeros((10, 5, nangles))

    # The angular scale is the same for all diffractograms
    for i in range(ndet + 1):
        diff[i, 0, :] = np.arange(angmin, angmax, angstep)

    # The Q-scale is the same for all diffractograms
    for i in range(ndet + 1):
        diff[i, 4, :] = ang2q(diff[i, 0, :], wlength=runInfo["wavelength"])

    # Number of numors to regroup and number of angles in one numor
    nbrNumors = len(numor[:, 0, 0])
    nbrData = len(numor[0, 0, :])

    # This variable will be used for normalisation
    # It is divided by 1000000 just to avoid large numbers
    totalmon = 0
    for num in range(nbrNumors):
        totalmon += head[num]["MonitorCnts"] / 1000000

    for num in range(nbrNumors):
        # Extract the values from the matrix numor using clearer names
        angle = numor[num, 0, :]
        count = numor[num, 1, :]
        error = numor[num, 2, :]
        cell = numor[num, 4, :]
        # Calculates the normalisation factor for the weighted sum
        mon = head[num]["MonitorCnts"] / 1000000 / totalmon
        for ang in range(nbrData):
            # Left side of the rectangle
            angle1 = angle[ang] - angular_step / 2.0
            ang1 = (angle1 - angmin) / angstep
            a1 = int(ang1)
            # Right side of the rectangle
            angle2 = angle[ang] + angular_step / 2.0
            ang2 = (angle2 - angmin) / angstep
            a2 = int(ang2)
            # Fraction that falls on each bin: left=1, right=2
            frac1 = 1.0 - (ang1 - int(ang1))
            frac2 = ang1 - int(ang1)
            # Checks if counts and errors are numbers
            # If NaN, it is not added to the histogram
            if (np.isnan(count[ang]) == False) and (
                np.isnan(error[ang]) == False):
                # The index 0 is used for the total diffractogram
                diff[0, 1, a1] += frac1 * count[ang] * mon
                diff[0, 1, a2] += frac2 * count[ang] * mon
                diff[0, 2, a1] += frac1 * error[ang] * mon
                diff[0, 2, a2] += frac2 * error[ang] * mon
                diff[0, 3, a1] += frac1 * mon
                diff[0, 3, a2] += frac2 * mon
                # Creates a histogram for individual detectors
                # det is an integer from 1 to 9
                det = int(cell[ang] / 1000)
                diff[det, 1, a1] += frac1 * count[ang] * mon
                diff[det, 1, a2] += frac2 * count[ang] * mon
                diff[det, 2, a1] += frac1 * error[ang] * mon
                diff[det, 2, a2] += frac2 * error[ang] * mon
                diff[det, 3, a1] += frac1 * mon
                diff[det, 3, a2] += frac2 * mon

    # Normalisation by the weighting factor of each bin
    for det in range(10):
        for i in range(len(diff[0, 1, :])):
            if (np.isnan(diff[0, 1, i]) == False) and (diff[det, 3, i] > 0):
                diff[det, 1, i] = diff[det, 1, i] / diff[det, 3, i]
                diff[det, 2, i] = diff[det, 2, i] / diff[det, 3, i]
            else:
                diff[det, 1, i] = "NaN"
                diff[det, 2, i] = "NaN"

    return diff
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def saveDiffAngle(diffA, head, runInfo):
    """
--------------------------------------------------------------------------------
    """
#--------1---------2---------3---------4---------5---------6---------7---------8
    # Loop to count the number of data
    ndata = 0
    for i in range(len(diffA[0, 0, :])):
        angle = diffA[0, 0, i]
        count = diffA[0, 1, i]
        error = diffA[0, 2, i]
        if (np.isnan(count * error) == False) and (count > 0):
            ndata += 1

    # Writing a file with CORRECT format
    file = runInfo["file"] + runInfo["ext"][1]
    with open(file, "w") as adatfile:
        adatfile.write("# " + file + "\n")
        adatfile.write("#Block  1\n")
        adatfile.write("#========\n")
        adatfile.write("#\n")
        adatfile.write("#Instrument:" + head[0]["label"][0:2] + "\n")
        adatfile.write("#User      :"+ head[0]["user"].strip(" ")+ "    exp_"
            + head[0]["proposal"].strip(" ")+ "/processed"+ "\n")
        adatfile.write("#Run number:            1\n")
        adatfile.write("#Spectrum  :            1\n")
        adatfile.write("#Title     :" + file + "\n")

        adatfile.write("#Run date  :" + runInfo["runDate"] + "\n")

        mini = runInfo["angular_scale"][0]
        maxi = runInfo["angular_scale"][1]
        step = runInfo["angular_scale"][2]
        bins = int((maxi - mini) / step) + 1
        adatfile.write(
            "#X caption : 2Theta (degrees) binned from {} to {} by {} ({} bins)".format(
                mini, maxi, step, bins)+ "\n")
        adatfile.write("#Y caption : Counts/monitor \n")
        adatfile.write("#Histogram :            F \n")
        adatfile.write("#Points    :         {}   ".format(ndata) + "\n")

        for i in range(len(diffA[0, 0, :])):
            angle = diffA[0, 0, i]
            count = diffA[0, 1, i]
            error = diffA[0, 2, i]
            if (np.isnan(count * error) == False) and (count > 0):
                adatfile.write(
                    "{:8.4f}     {:12.8f}     {:12.8f} \n".format(
                        angle, count, error))
    print(4 * " "+ "File "+ runInfo["file"]+ runInfo["ext"][1]
        + " (CORRECT format, in angle scale)")

    listNum = " "
    for i in range(len(runInfo["numorLst"])):
        listNum += runInfo["numorLst"][i] + " "

    # Writing the data as 9 individual detectors (format reg from D4)
    file = runInfo["file"] + runInfo["ext"][0]
    with open(file, "w") as regfile:
        regfile.write("# " + file + "\n")
        regfile.write("# Equivalent command line: \n")
        regfile.write("#     " + runInfo["d4creg_cl"] + "\n")
        regfile.write("# Efficiency file: " + runInfo["efffile"] + "\n")
        regfile.write("# Shift file: " + runInfo["decfile"] + "\n")
        regfile.write("# Sample: " + head[0]["subtitle"] + "\n")
        regfile.write("# User: " + head[0]["user"].strip(" ") + "\n")
        regfile.write("# Local contact: " + head[0]["LC"].strip(" ") + "\n")
        regfile.write("# Proposal: " + head[0]["proposal"].strip(" ") + "\n")
        regfile.write("# Run date  :" + runInfo["runDate"] + "\n")
        regfile.write(
            "# Requested numors: {}, {} numors included \n".format(
                listNum, runInfo["nbrNumors"]))
        regfile.write(
            "# Normalisation type: " + runInfo["normalisation"][0] + "\n")
        regfile.write("# Zero-angle = " + str(runInfo["twotheta0"]) + " deg\n")
        regfile.write(
            "# Wavelength = " + str(runInfo["wavelength"]) + " deg\n")
        regfile.write(
            "# Monitor dead-time = " + str(runInfo["dead_time"][0]) + " s\n")
        regfile.write(
            "# Detector dead-time = " + str(runInfo["dead_time"][1]) + " s\n")
        regfile.write("# 2theta = 2theta.raw - Zero-angle + shifts \n")
        regfile.write(
            "# Binned on angle (from {} to {} in steps of {}, deg), but not on Q \n".format(
                *runInfo["angular_scale"]))
        regfile.write(
            "#     Q = 4pi/{}Å * sin(angle/2) \n".format(
                str(runInfo["wavelength"])))

        for det in range(1, 10):
            regfile.write("# ----------- \n")
            regfile.write("# Detector " + str(det) + "\n")
            regfile.write("# ----------- \n")
            regfile.write(
                "# Angle(deg)     Counts           Error            Q(1/Å)\n"
            )
            for i in range(len(diffA[0, 0, :])):
                angle = diffA[det, 0, i]
                count = diffA[det, 1, i]
                error = diffA[det, 2, i]
                qval = diffA[det, 4, i]
                if (np.isnan(count * error) == False) and (count > 0):
                    regfile.write(
                        "{:8.4f}     {:12.8f}     {:12.8f}     {:8.4f} \n".format(
                            angle, count, error, qval))

    print("    File "+ runInfo["file"]+ runInfo["ext"][0]
        + " (D4 format, in angle and Q-scale)")
    return
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def saveDiffQ(diffQ, head, runInfo):
    """
--------------------------------------------------------------------------------
    """
#--------1---------2---------3---------4---------5---------6---------7---------8
    # Loop to count the number of data
    ndata = 0
    for i in range(len(diffQ)):
        #        q = diffQ[i,0]
        count = diffQ[i, 1]
        error = diffQ[i, 2]
        if (np.isnan(count * error) == False) and (count > 0): ndata += 1

    # Writing a file with CORRECT format
    file = runInfo["file"] + runInfo["ext"][2]
    with open(file, "w") as qdatfile:
        qdatfile.write("# " + file + "\n")
        qdatfile.write("#Block  1\n")
        qdatfile.write("#========\n")
        qdatfile.write("#\n")
        qdatfile.write("#Instrument:" + head[0]["label"][0:2] + "\n")
        qdatfile.write("#User      :"+ head[0]["user"].strip(" ")
            + "    exp_"+ head[0]["proposal"].strip(" ")+ "/processed"+ "\n")
        qdatfile.write("#Run number:            1\n")
        qdatfile.write("#Spectrum  :            1\n")
        qdatfile.write("#Title     :" + file + "\n")
        qdatfile.write("#Run date  :" + runInfo["runDate"] + "\n")
        mini = runInfo["q_scale"][0]
        maxi = runInfo["q_scale"][1]
        step = runInfo["q_scale"][2]
        bins = int((maxi - mini) / step) + 1
        qdatfile.write(
            "#X caption : Q (1/Å) binned from {:4.3f} to {:4.3f} by {:4.3f} ({} bins)".format(
                mini, maxi, step, bins)+ "\n")
        qdatfile.write("#Y caption : Counts/monitor\n")
        qdatfile.write("#Histogram :            F\n")
        qdatfile.write("#Points    :         {}   ".format(ndata) + "\n")

        for i in range(len(diffQ)):
            q = diffQ[i, 0]
            count = diffQ[i, 1]
            error = diffQ[i, 2]
            if (np.isnan(count * error) == False) and (count > 0):
                qdatfile.write(
                    "{:8.4f}     {:12.8f}     {:12.8f} \n".format(
                        q, count, error))
    print("    File "+ runInfo["file"]+ runInfo["ext"][2]
        + " (CORRECT format, in Q scale)")
    return
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def defColors():
    """
--------------------------------------------------------------------------------
    """
    # names = list(mcolors.BASE_COLORS)
    # print (names)
    # names = list(mcolors.TABLEAU_COLORS)
    # print (names)
    # names = list(mcolors.CSS4_COLORS)
    # print (names)

#--------1---------2---------3---------4---------5---------6---------7---------8
    colors = []
    colors.append(mcolors.CSS4_COLORS["red"])         #  0
    colors.append(mcolors.CSS4_COLORS["blue"])        #  1
    colors.append(mcolors.CSS4_COLORS["green"])       #  2
    colors.append(mcolors.CSS4_COLORS["black"])       #  3
    colors.append(mcolors.CSS4_COLORS["cyan"])        #  4
    colors.append(mcolors.CSS4_COLORS["magenta"])     #  5
    colors.append(mcolors.CSS4_COLORS["gold"])        #  6
    colors.append(mcolors.CSS4_COLORS["orange"])      #  7
    colors.append(mcolors.CSS4_COLORS["brown"])       #  8
    colors.append(mcolors.CSS4_COLORS["gray"])        #  9
    colors.append(mcolors.CSS4_COLORS["silver"])      # 10
    colors.append(mcolors.CSS4_COLORS["pink"])        # 11
    colors.append(mcolors.CSS4_COLORS["purple"])      # 12
    colors.append(mcolors.CSS4_COLORS["navy"])        # 13
    colors.append(mcolors.CSS4_COLORS["teal"])        # 14
    colors.append(mcolors.CSS4_COLORS["lime"])        # 15
    colors.append(mcolors.CSS4_COLORS["olive"])       # 16
    colors.append(mcolors.CSS4_COLORS["salmon"])      # 17
    colors.append(mcolors.CSS4_COLORS["maroon"])      # 18
    colors.append(mcolors.CSS4_COLORS["chartreuse"])  # 19
    return colors
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def makePlotsA(runInfo, diffA, head, numor):
    """
--------------------------------------------------------------------------------
    """
    colors = defColors()
    # Plot the individual numors
    plt.figure(figsize=(9, 5))
    plt.title("Individual numors ({} in total) for sample {}".format(
            runInfo["nbrNumors"], runInfo["file"]))

    amin, amax = runInfo["angular_scale"][0], runInfo["angular_scale"][1]
    ymin, ymax = 0.9*np.nanmin(diffA[0,1,:]), 1.1*np.nanmax(diffA[0,1,:])
    plt.axis([amin, amax, ymin, ymax])
    plt.xlabel("Scattering angle (degrees)")
    if runInfo["normalisation"][0] == "monitor":
        plt.ylabel(
            "Counts/(" + str(runInfo["normalisation"][2]) + " monitor counts)")
    else:
        plt.ylabel("Counts/(" + str(runInfo["normalisation"][1]) + " seconds)")
    for i in range(numor.shape[0]):
        plt.plot(numor[i,0,:],numor[i,1,:],
                 colors[i % 20],label=head[i]["numor"])
    plt.grid(True)
    plt.show()

    # Plot the individual detectors
    plt.figure(figsize=(9, 6))
    plt.title("Individual detectors for " + runInfo["file"])
    amin, amax = runInfo["angular_scale"][0], runInfo["angular_scale"][1]
    ymin, ymax = 0.9*np.nanmin(diffA[0,1,:]), 1.1*np.nanmax(diffA[0,1,:])
    plt.axis([amin, amax, ymin, ymax])
    plt.xlabel("Scattering angle (degrees)")
    if runInfo["normalisation"][0] == "monitor":
        plt.ylabel(
            "Counts/(" + str(runInfo["normalisation"][2]) + " monitor counts)")
    else:
        plt.ylabel("Counts/(" + str(runInfo["normalisation"][1]) + " seconds)")
    for i in range(1, 10):
        plt.plot(diffA[i,0,:],diffA[i,1,:],
            colors[i],label="Detector " + str(i))
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

    # Plot the final diffractogram
    plt.figure(figsize=(9, 6))
    plt.title("Diffractogram for " + runInfo["file"])
    amin, amax = runInfo["angular_scale"][0], runInfo["angular_scale"][1]
    ymin, ymax = 0.9*np.nanmin(diffA[0,1,:]), 1.1*np.nanmax(diffA[0, 1, :])
    plt.axis([amin, amax, ymin, ymax])
    plt.xlabel("Scattering angle (degrees)")
    if runInfo["normalisation"][0] == "monitor":
        plt.ylabel(
            "Counts/(" + str(runInfo["normalisation"][2]) + " monitor counts)")
    else:
        plt.ylabel("Counts/(" + str(runInfo["normalisation"][1]) + " seconds)")
    #    linestyle = 'solid','dotted','dashed','dashdot'
    #    marker = 'o', 'v', '^','+','x'
    plt.plot(diffA[0,0,:],diffA[0,1,:],
        color=colors[0],linestyle="solid",label=runInfo["file"])
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    return
#--------1---------2---------3---------4---------5---------6---------7---------8

#--------1---------2---------3---------4---------5---------6---------7---------8
def makePlotsQ(runInfo, diffQbin, head, numor):
    """
--------------------------------------------------------------------------------
    """
    colors = defColors()

    # Plot the final diffractogram
    plt.figure(figsize=(9, 6))
    plt.title("Diffractogram for " + runInfo["file"])
    qmin, qmax = runInfo["q_scale"][0], runInfo["q_scale"][1]
    ymin, ymax = 0.9*np.nanmin(diffQbin[:,1]), 1.1*np.nanmax(diffQbin[:, 1])
    plt.axis([qmin, qmax, ymin, ymax])
    plt.xlabel("Momentum transfer (1/Å)")
    if runInfo["normalisation"][0] == "monitor":
        plt.ylabel(
            "Counts/(" + str(runInfo["normalisation"][2]) + " monitor counts)")
    else:
        plt.ylabel("Counts/(" + str(runInfo["normalisation"][1]) + " seconds)")
    #    linestyle = 'solid','dotted','dashed','dashdot'
    #    marker = 'o', 'v', '^','+','x'
    plt.plot(diffQbin[:, 0],diffQbin[:, 1],
        color=colors[0],linestyle="solid",label=runInfo["file"])
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    return
#--------1---------2---------3---------4---------5---------6---------7---------8


#--------1---------2---------3---------4---------5---------6---------7---------8
def d4creg(runInfo):
    """
--------------------------------------------------------------------------------
    """
    runInfo["runDate"] = getDate()  # Today's date

    numor, head = getNumors(runInfo)

    runInfo["nbrNumors"] = len(head)

    # Creating and printing the equivalent d4creg command line
    d4creg_cl = "d4creg"
    d4creg_cl += " -x " + str(runInfo["angular_scale"][0]) + " "
    d4creg_cl += str(runInfo["angular_scale"][1]) + " "
    d4creg_cl += str(runInfo["angular_scale"][2])
    d4creg_cl += " -z " + str(runInfo["twotheta0"])
    d4creg_cl += " -w " + str(runInfo["wavelength"])
    d4creg_cl += " -o " + runInfo["file"] + ".reg "
    d4creg_cl += head[0]["numor"] + " " + head[-1]["numor"]
    runInfo["d4creg_cl"] = d4creg_cl

    print()
    print("Equivalent d4creg command line")
    print(d4creg_cl)
    print()
    print(" - Efficiency: ", runInfo["efffile"])
    print(" - Det shifts: ", runInfo["decfile"])
    if runInfo["normalisation"][0] == "monitor":
        print(
            " - Normalisation to {1} of {0} counts".format(
                runInfo["normalisation"][0], runInfo["normalisation"][2]
            )
        )
    else:
        print(
            " - Normalisation to {1} s of counting {0}".format(
                runInfo["normalisation"][0], runInfo["normalisation"][1]
            )
        )
    print(" - Number of numors: ", runInfo["nbrNumors"])

    # Creating the diffactogram in angular scale
    diffA = getDiffA(numor, head, runInfo)

    # Saving the output files in angular scale
    print()
    print("Output:")
    saveDiffAngle(diffA, head, runInfo)

    if runInfo["plotDiff"] == 1:
        makePlotsA(runInfo, diffA, head, numor)

    qmin = runInfo["q_scale"][0]
    qmax = runInfo["q_scale"][1]
    qstep = runInfo["q_scale"][2]

    dataQ = np.zeros((diffA.shape[2], 3))

    ang = diffA[0, 0, :]
    dataQ[:, 0] = ang2q(ang, wlength=runInfo["wavelength"])

    for i in range(diffA.shape[2]):
        for j in range(1, 3):
            dataQ[i, j] = diffA[0, j, i]

    diffQbin = rebin(0.125, runInfo["wavelength"], dataQ, qmin, qmax, qstep)

    for i in range(len(diffQbin)):
        if diffQbin[i, 1] <= 0.0:
            for j in range(3):
                diffQbin[i, j] = "NaN"
                diffQbin[i - 1, j] = "NaN"

    saveDiffQ(diffQbin, head, runInfo)

    if runInfo["plotDiff"] == 1:
        makePlotsQ(runInfo, diffQbin, head, numor)

# Here change the name of the logfile

    # Specify the current path of the file you want to rename
    current_file_path = '../logfiles/d4creg.log'

    # Specify the new name and path for the file
    new_file_name = '../logfiles/'+runInfo["logfile"]+'.log'

    # Rename the file
    try:
        os.rename(current_file_path, new_file_name)
        print(f"File '{current_file_path}' renamed to '{new_file_name}' successfully.")
    except FileNotFoundError:
        print(f"File '{current_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


    return
#--------1---------2---------3---------4---------5---------6---------7---------8

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#
#  Fitting Nickel Data
#
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################



def find_minimum_within_range(x, y, x_min, x_max):
    # Mask the data within the specified x range
    mask = (x >= x_min) & (x <= x_max)
    x_range = x[mask]
    y_range = y[mask]

    # Find the index of the minimum y value within the specified x range
    min_index = np.argmin(y_range)

    # Get the x position and minimum y value
    min_x_position = x_range[min_index]
    min_y_value = y_range[min_index]

    return min_x_position, min_y_value


def find_peaks_in_range(x, y, x_min, x_max):
    # Mask the data within the specified x range
    mask = (x >= x_min) & (x <= x_max)
    x_range = x[mask]
    y_range = y[mask]

    # Find peaks within the specified x range
    peaks, _ = find_peaks(y_range)
    _ , ymin = find_minimum_within_range(x, y, x_min, x_max)

    # Get the x positions and maximum values of the peaks within the range
    peak_x_positions = x_range[peaks]
    peak_maximum_values = y_range[peaks]/ymin

    mask = (peak_maximum_values > 3.5)
    peak_x = peak_x_positions[mask]
    peak_y = peak_maximum_values[mask]

    return peak_x, peak_y


#--------1---------2---------3---------4---------5---------6---------7---------
class DataXYE():
    """
Type: Class

Object:
    To read data files in ASCII format, two (X Y) or three (X Y E) columns

Input:
    filename: (string) Filename of the file containing the data.

Output:
    An instance created with the following attributes and methods.

    self.filename: (string) Filename of the input file
    self.basename: (string) File basename (what it is before extention's dot)
    self.ext: (string) File extension (without the dot)
    self.x: Abscissas
    self.y: Ordinates
    self.e: Errors (or 3rd coordinate). Returns -1 for 2-column files.
    self.head: List of strings with each line in the file header.

    self.xave: Mean value of the abscissas
    self.yave: Mean value of the ordinates
    self.eave: Mean value of the errors

    self.xmin: Minimum value of the abscissas
    self.ymin: Minimum value of the ordinates
    self.emin: Minimum value of the errors

    self.xmax: Maximum value of the abscissas
    self.ymax: Maximum value of the ordinates
    self.emax: Maximum value of the errors

    self.plot(): Makes a simple plot of y coordinate
    self.show(): Shows the data on screen (as a 3-column table)
    self.header(): Prints the header of the file

Author: Gabriel Cuello
Created: 29/12/2022
Modified:
#--------1---------2---------3---------4---------5---------6---------7---------
    """

    def __init__(self,filename):
        """
Type: Main function of the Class DataXYE
    The file is read and the attributes are defined here.

Input:
    filename: (string) The filename of the file containing the data.

Output:
    The attributes that can be accessed by the instances.
    See the help of this Class for a complete list of attributes.

Author: Gabriel Cuello
Created: 29/12/2022
Modified:
#--------1---------2---------3---------4---------5---------6---------7---------
        """
        self.filename = filename
        self.basename = os.path.splitext(filename)[0]
        self.ext = os.path.splitext(filename)[1][1:] # Exclude 1st character to avoid the dot
        self.x = []
        self.y = []
        self.e = []
        self.head = []

        data = open(self.filename,'r')
        lines = data.readlines()
        for dataline in lines:
            row = dataline.strip(' ')[:-1]
            if len(row)>0:  # Only the non empty lines are treated
                if row[0] == "#" or row[0] == "!":
                    self.head.append(row)
                else:
                    columns = row.split()   # This method split the line using the spaces
                    if len(columns) == 2:
                        self.x.append(float(columns[0]))
                        self.y.append(float(columns[1]))
                        self.e.append(-1.0)
                    elif len(columns) == 3:
                        self.x.append(float(columns[0]))
                        self.y.append(float(columns[1]))
                        self.e.append(float(columns[2]))
                    else:
                        print ("Wrong file format")
                        sys.exit()
        data.close()
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.e = np.array(self.e)
        self.xave = st.mean(self.x)
        self.xmin = min(self.x)
        self.xmax = max(self.x)
        self.yave = st.mean(self.y)
        self.ymin = min(self.y)
        self.ymax = max(self.y)
        self.eave = st.mean(self.e)
        self.emin = min(self.e)
        self.emax = max(self.e)
        self.peaks_x, self.peaks_y = find_peaks_in_range(self.x, self.y, 5.0, 40.0)
        self.xminr, self.yminr = find_minimum_within_range(self.x, self.y, 5.0, 40.0)

    def plot(self,file_format=0,xmin=None,xmax=None, ymin=None, ymax=None):
        """
Type: Method in DataXYE class

Object:
    To make a simple plot of the ordinates as function of abscissas.
    To produce a file with the plot.

Input:
    xmin,xmax: Minimum and maximum values of the x-axis (float, optional)
    ymin,ymax: Minimum and maximum values of the y-axis (float, optional)
    file_format: A string that defines the format (and extension) of the ouput
                 file (string, optional)

Output:
    A simple plot on the screen.
    A file with the plot in a graphical file.

Remarks:
  * Several formats are possible for the output file. The kind of file is
    defined by the input parameter file_format, which can must take one
    of the following values: 'png','pdf','jpg','tiff','svg','jpeg','ps','eps'.
    If this paramteter is not present, it takes the default value 0 and no
    output file is created.

  * The output file has the same basename as the input file, but the extension
    corresponding to chosen format.

  * The limits of the axes are optional. Their default value is None, which
    will produce a plot with automatic limits.

Author: Gabriel Cuello
Created: 29/12/2022
Modified:
#--------1---------2---------3---------4---------5---------6---------7---------
        """
        A_ext = ['adat', 'Adat', 'reg']
        Q_ext = ['qdat', 'Qdat', 'Qreg', 'soq', 'SoQ']
        R_ext = ['pcf', 'pdf', 'tor', 'rdf', 'run']

        plt.figure(figsize=(9,6))

        plt.plot(self.x,self.y, 'r-+',label=self.filename)

        plt.legend(loc='best')
        plt.title('Data in ' + self.filename)
        plt.xlabel('Abscissa')
        if self.ext in A_ext:
            plt.xlabel(r'$2\theta$ (˚)')
        elif self.ext in Q_ext:
            plt.xlabel(r'$Q$ (Å${-1}$)')
        elif self.ext in R_ext:
            plt.xlabel('$R$ (Å)')
        plt.ylabel('Intensity (arb. units)')
        plt.axis([xmin, xmax, ymin, ymax])
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        if file_format in ['png','pdf','jpg','tiff','svg','jpeg','ps','eps']:
            file_fig = '../regdata/'+self.basename+'.'+file_format
            plt.savefig(file_fig, format=file_format)
            print("Figure saved on {}".format(file_fig))

    def show(self):
        """
Type: Method in DataXYE class

Object:
    To show the data on the screen.

Input: None

Output:
    Print out of data on the screen in a 3-column table.

Author: Gabriel Cuello
Created: 29/12/2022
Modified:
#--------1---------2---------3---------4---------5---------6---------7---------
        """
        A_ext = ['adat', 'Adat', 'reg']
        Q_ext = ['qdat', 'Qdat', 'Qreg', 'soq', 'SoQ']
        R_ext = ['pcf', 'pdf', 'tor', 'rdf', 'run']

        if self.ext in A_ext:
            print(f"{'Ang':>6}{'Intensity':>12}{'Error':>12}")
        elif self.ext in Q_ext:
            print(f"{'Q':>6}{'Intensity':>12}{'Error':>12}")
        elif self.ext in R_ext:
            print(f"{'R':>6}{'Intensity':>12}{'Ignore':>12}")
        else:
            print(f"{'x':>6}{'y':>12}{'e':>12}")
        for i in range(len(self.x)):
            print(f"{self.x[i]:>6}{self.y[i]:>12}{self.e[i]:>12}")

    def header(self,lines=1000):
        """
Type: Method in DataXYE class

Object:
    To print the file header on the screen.

Input:
    lines: number of lines to be printed

Output:
    Print out of file heading on the screen.

Remarks:
  * The default value for the input parameter lines is 1000. But this method
    will print only the number of lines in the header, unless this number is
    greater than 1000.

Author: Gabriel Cuello
Created: 29/12/2022
Modified:
#--------1---------2---------3---------4---------5---------6---------7---------
        """
        for i in range(min(lines,len(self.head))):
            print(f"{i+1:>3} {self.head[i]}")

#--------1---------2---------3---------4---------5---------6---------7---------


def setting_model_d4nifit(nifile,zac_ini=0.0):

# Reading the data file and creating an instance of the class DataXYE
    nickel = DataXYE(nifile)

# Plot the not yet normalised data
    nickel.plot()

# Because the structure of Ni is fcc (with a lattice parameter of a = 3.52024 Å), the
# first peak corresponds to the reflection (111):
#         2 * a * sin (2theta/2) = sqrt(h**2+k**2+l**2) * lambda
# where 2theta is the scattering angle of the observed first peak.
# Thus, a nominal wavelength (and initial value for the fitting process) can easily
# be estimated:
#     lambda = 2 * a * sin(2theta/2) / sqrt(3)
#
# The instance has a property called peaks_x, which is a list of the peak positions
# found in the range 5-40 degrees. Thus, the angular position of the first peak is
# peaks_x[0]

    lattice = 3.52024        # Lattice parameter of fcc Ni at room temperature
    wl_ini=2*lattice/np.sqrt(3.0)*np.sin(nickel.peaks_x[0]*np.pi/360.0)
    print (f"Automatically calculated initial wavelength = {wl_ini:.3f} Å")
   # print (nickel.peaks_x,nickel.peaks_y)



# There is a flat valley between reflections (200) and (220). For normalisation
# purposes (with the idea of having "standard" plots), the intensity in that flat
# valley is used to normalise the data. The middle point between the 2nd peak (200)
# and 3rd peak (220) is chosen for that normalisation. The positions of these peaks
# are peaks_x[1] and peaks_x[2], respectively.

    flatValley_x = int((nickel.peaks_x[1]+nickel.peaks_x[2])/2)

# Find the index where x equals the flatValley_x
#    index = np.where(nickel.x == flatValley_x )
    for i in range(len(nickel.x)):
        if nickel.x[i] > flatValley_x:
            index = i
            break
    flatValley_y = nickel.y[index]

# Check if the flatValley_x exists in x
    # if index[0].size > 0:
    #     # Get the corresponding value of y using the index
    #     flatValley_y = nickel.y[index][0]
    #     print(f"Normalised data using the intensity at {flatValley_x:.0f} degrees: {flatValley_y:.2f}")
    # else:
    #     print(f"2theta = {flatValley_x:.0f} is not in the array of angles.")

# Normalise the y-scale
    nickel.y = nickel.y/flatValley_y
    nickel.ymax = nickel.ymax/flatValley_y

# Plot the already normalised data
    nickel.plot()

# Using the nominal wavelength, some wavelength dependent parameters are initialised:
# axes:   these are the limits of the plots showing the detailed region of fitting
# limits: these are the limits used in the fitting
# Note that by chance 93 times the wavelength is the good upper limit in angle.
    axes =   [0.5*flatValley_x, 93 * wl_ini,    -0.1*nickel.ymax, 1.1*nickel.ymax]
    limits = [0.5*flatValley_x, 93 * wl_ini,                   0, 1.1*nickel.ymax]

# Here the coordinates of text blocks in plots are defined.
# res: block with the results (wavelength and zero)
# res: block with the list of reflections
# fit: block with the information about the fit (iterations and other)
    xy_res = [0.81 * flatValley_x,  0.90 * max(nickel.y)]
    xy_ref = [1.60 * flatValley_x,  0.40 * max(nickel.y)]
    xy_fit = [1.85 * flatValley_x,  0.75 * max(nickel.y)]

# FPAarea is the area of the first peak. Approximating this area by
# A = 2 * sigma * I_max, where sigma is about 0.25, then A = 0.5 * I_max
    FPArea = 0.5 * (nickel.ymax - 1)
#    print ('Area =', FPArea)

# Setting the initial values for all parameters:
# - the 3 first parameters are for the polynomial
# - then wl and zac
# - then the 10 areas, the 10 asymmetric factors and the 10 sigmas (Gaussian functions)
# Note that the initial centroids are not setted because they are calculated from the
# structure of the fcc lattice, for the initial wavelength and zero angle correction, and
# the known lattice constant for Ni (a = 3.52024 Å).
    initial = [1,0.0007*FPArea,0.0,wl_ini,zac_ini,
               1.00*FPArea, 0.56*FPArea, 0.54*FPArea, 0.76*FPArea, 0.24*FPArea,
               0.12*FPArea, 0.42*FPArea, 0.40*FPArea, 0.32*FPArea, 0.37*FPArea,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4]

    # List of the 10 planes that will be used in the fitting
    planes = ['(111)','(200)','(220)','(311)','(222)','(400)','(331)','(420)','(422)','(333)']


# Calculates the first ten reflections
    refl = d4.reflections_fcc(wl_ini, zac_ini, lattice=3.52024)[:10]
    model = d4.niPeaks10(nickel.x,*initial)
    #initial_text = "lambda = {:.6f} Å \n2theta0 = {:.6f} deg".format(wl_ini,zac_ini)

# Plot the figure with the starting point for the fitting
    plt.figure(figsize=(9,6))

    plt.plot(nickel.x, model,          'r-',  label='Initial model')
    plt.plot(nickel.x, nickel.y,       'b-+', label='Ni powder data')
    plt.plot(nickel.x, nickel.y-model, 'g-+', label='Residuals')

    plt.legend(loc='best')
    plt.title('Nickel powder diffractogram: Starting point')
    plt.xlabel(r'$2\theta$ (˚)')
    plt.ylabel('Intensity (arb. units)')

    wlength = r"$\lambda = {:.3f}$ Å".format(wl_ini)
    zac = r"$2\theta_0 = {:.3f}$˚".format(zac_ini)

    plt.text(xy_res[0],xy_res[1],
             s=wlength+'\n'+zac, bbox=dict(facecolor='white', alpha=1.0, edgecolor='black'))

    planes_fit = ''
    for i in range(10):
        planes_fit += '{}: {:.1f}˚\n'.format(planes[i], refl[i])
    plt.text(xy_ref[0],xy_ref[1],
             s=planes_fit[:-1], bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    plt.axis(axes)
    plt.grid(True)
    plt.tight_layout()
    return axes,limits,xy_res,xy_ref,xy_fit,planes,initial,nickel

def fitting_model_d4nifit(nickel,limits,initial):
#         """
# Type: function

# Object:
#     Defines the fitting model for the fit of Ni diffractogram for wavelength
#     calibration nad perform the fitting

# Input:
#     nickel: An instance of DataXYE class with the data
#     limits: List with for values corresponding to the fitting box
#     initial: Initial values of the fitting parameters

# Output:
#     An object containing the results of the fitting process.

# Remarks:
#   * In thsi function you can choose the parameters to be fitted.

# Author: Gabriel Cuello
# Created: 29/12/2022
# Modified:
# #--------1---------2---------3---------4---------5---------6---------7---------
#         """
    xfit = nickel.x
    yfit = nickel.y
    efit = nickel.e

    xfit,yfit,efit = d4.fittingRange(limits[0],limits[1],limits[2],limits[3],nickel.x,nickel.y,nickel.e)

    gmodel = lm.Model(d4.niPeaks10)
    gmodel.set_param_hint('I0',vary=True)
    gmodel.set_param_hint('slope',vary=True)
    gmodel.set_param_hint('quad',vary=False)
    gmodel.set_param_hint('wavelength',vary=True)
    gmodel.set_param_hint('twotheta0',vary=True)
    gmodel.set_param_hint('A0',vary=True)
    gmodel.set_param_hint('A1',vary=True)
    gmodel.set_param_hint('A2',vary=True)
    gmodel.set_param_hint('A3',vary=True)
    gmodel.set_param_hint('A4',vary=True)
    gmodel.set_param_hint('A5',vary=True)
    gmodel.set_param_hint('A6',vary=True)
    gmodel.set_param_hint('A7',vary=True)
    gmodel.set_param_hint('A8',vary=True)
    gmodel.set_param_hint('A9',vary=True)
    gmodel.set_param_hint('G0',vary=False)
    gmodel.set_param_hint('G1',vary=False)
    gmodel.set_param_hint('G2',vary=False)
    gmodel.set_param_hint('G3',vary=False)
    gmodel.set_param_hint('G4',vary=False)
    gmodel.set_param_hint('G5',vary=False)
    gmodel.set_param_hint('G6',vary=False)
    gmodel.set_param_hint('G7',vary=False)
    gmodel.set_param_hint('G8',vary=False)
    gmodel.set_param_hint('G9',vary=False)
    gmodel.set_param_hint('S0',vary=True)
    gmodel.set_param_hint('S1',vary=True)
    gmodel.set_param_hint('S2',vary=True)
    gmodel.set_param_hint('S3',vary=True)
    gmodel.set_param_hint('S4',vary=True)
    gmodel.set_param_hint('S5',vary=True)
    gmodel.set_param_hint('S6',vary=True)
    gmodel.set_param_hint('S7',vary=True)
    gmodel.set_param_hint('S8',vary=True)
    gmodel.set_param_hint('S9',vary=True)

    # Here the fit is performed
    result = gmodel.fit(yfit,x=xfit, I0=initial[0], slope=initial[1], quad=initial[2],
                        wavelength=initial[3], twotheta0=initial[4],
                        A0=initial[ 5],A1=initial[ 6],A2=initial[ 6],A3=initial[ 7],A4=initial[ 8],
                        A5=initial[10],A6=initial[11],A7=initial[12],A8=initial[13],A9=initial[14],
                        G0=initial[15],G1=initial[16],G2=initial[17],G3=initial[18],G4=initial[19],
                        G5=initial[20],G6=initial[21],G7=initial[22],G8=initial[23],G9=initial[24],
                        S0=initial[25],S1=initial[26],S2=initial[27],S3=initial[28],S4=initial[29],
                        S5=initial[30],S6=initial[31],S7=initial[32],S8=initial[33],S9=initial[34])
    return result

def showing_results_d4nifit(result,nickel,axes,limits,xy_res,xy_ref,xy_fit,planes):
    print()
    print('Results of fitting the nickel powder sample')

    nickel_table = {}
    nickel_par = []

    for param in result.params.values():
        if param.value != 0:
            relative = abs(param.stderr / param.value)
        else:
            relative = 0.0
        nickel_table[param.name] = [param.value, param.stderr, relative]
        nickel_par.append(param.value)
        if param.name == 'wavelength':
            wlength = r"$\lambda = ({:.6f} \pm {:.6f})$ Å".format(param.value, param.stderr)
        if param.name == 'twotheta0':
            zac = r"$2\theta_0 = ({:.6f} \pm {:.6f})$˚".format(param.value, param.stderr)

    model = d4.niPeaks10(nickel.x,*nickel_par)
    refl = d4.reflections_fcc(nickel_par[3], nickel_par[4], lattice=3.52024)[:10]

    print( f"{result.params['I0'].name:>10} = \
          {nickel_table['I0'][0]:>9f} ± {nickel_table['I0'][1]:>9f}         {nickel_table['I0'][2]:>7.3%}")

    print( f"{result.params['slope'].name:>10} = \
          {nickel_table['slope'][0]:>9f} ± {nickel_table['slope'][1]:>9f}         {nickel_table['slope'][2]:>7.3%}")

    print( f"{result.params['quad'].name:>10} = \
          {nickel_table['quad'][0]:>9f} ± {nickel_table['quad'][1]:>9f}         {nickel_table['quad'][2]:>7.3%}")

    print( f"{result.params['wavelength'].name:>10} = \
          ({nickel_table['wavelength'][0]:>8f} ± {nickel_table['wavelength'][1]:>9f}) Å      {nickel_table['wavelength'][2]:>7.3%}")

    print( f"{result.params['twotheta0'].name:>10} = \
          ({nickel_table['twotheta0'][0]:>8.5f} ± {nickel_table['twotheta0'][1]:>9f})˚       {nickel_table['twotheta0'][2]:>7.3%}")


    print(103*'-')
    print( f"|{'Plane':>6s} | {'Center':>7s}  | {'Area        ':>20s}  |  {'%   ':>8s} ||\
           {'Sigma        ':>19s}  |   {'%   ':>8s} |" )
    print(103*'-')
    print( f"|{planes[0]:>6s} | {refl[0]:>8.5f} | {nickel_table['A0'][0]:>8.5f} ± {nickel_table['A0'][1]:>9f}  |  {nickel_table['A0'][2]:>8.3%} ||\
          {nickel_table['S0'][0]:>8.5f} ± {nickel_table['S0'][1]:>9f}  |  {nickel_table['S0'][2]:>9.3%} |" )

    print( f"|{planes[1]:>6s} | {refl[1]:>8.5f} | {nickel_table['A1'][0]:>8.5f} ± {nickel_table['A1'][1]:>9f}  |  {nickel_table['A1'][2]:>8.3%} ||\
          {nickel_table['S1'][0]:>8.5f} ± {nickel_table['S1'][1]:>9f}  |  {nickel_table['S1'][2]:>9.3%} |" )
    print( f"|{planes[2]:>6s} | {refl[2]:>8.5f} | {nickel_table['A2'][0]:>8.5f} ± {nickel_table['A2'][1]:>9f}  |  {nickel_table['A2'][2]:>8.3%} ||\
          {nickel_table['S2'][0]:>8.5f} ± {nickel_table['S2'][1]:>9f}  |  {nickel_table['S2'][2]:>9.3%} |" )
    print( f"|{planes[3]:>6s} | {refl[3]:>8.5f} | {nickel_table['A3'][0]:>8.5f} ± {nickel_table['A3'][1]:>9f}  |  {nickel_table['A3'][2]:>8.3%} ||\
          {nickel_table['S3'][0]:>8.5f} ± {nickel_table['S3'][1]:>9f}  |  {nickel_table['S3'][2]:>9.3%} |" )
    print( f"|{planes[4]:>6s} | {refl[4]:>8.5f} | {nickel_table['A4'][0]:>8.5f} ± {nickel_table['A4'][1]:>9f}  |  {nickel_table['A4'][2]:>8.3%} ||\
          {nickel_table['S4'][0]:>8.5f} ± {nickel_table['S4'][1]:>9f}  |  {nickel_table['S4'][2]:>9.3%} |" )
    print( f"|{planes[5]:>6s} | {refl[5]:>8.5f} | {nickel_table['A5'][0]:>8.5f} ± {nickel_table['A5'][1]:>9f}  |  {nickel_table['A5'][2]:>8.3%} ||\
          {nickel_table['S5'][0]:>8.5f} ± {nickel_table['S5'][1]:>9f}  |  {nickel_table['S5'][2]:>9.3%} |" )
    print( f"|{planes[6]:>6s} | {refl[6]:>8.5f} | {nickel_table['A6'][0]:>8.5f} ± {nickel_table['A6'][1]:>9f}  |  {nickel_table['A6'][2]:>8.3%} ||\
          {nickel_table['S6'][0]:>8.5f} ± {nickel_table['S6'][1]:>9f}  |  {nickel_table['S6'][2]:>9.3%} |" )
    print( f"|{planes[7]:>6s} | {refl[7]:>8.5f} | {nickel_table['A7'][0]:>8.5f} ± {nickel_table['A7'][1]:>9f}  |  {nickel_table['A7'][2]:>8.3%} ||\
          {nickel_table['S7'][0]:>8.5f} ± {nickel_table['S7'][1]:>9f}  |  {nickel_table['S7'][2]:>9.3%} |" )
    print( f"|{planes[8]:>6s} | {refl[8]:>8.5f} | {nickel_table['A8'][0]:>8.5f} ± {nickel_table['A8'][1]:>9f}  |  {nickel_table['A8'][2]:>8.3%} ||\
          {nickel_table['S8'][0]:>8.5f} ± {nickel_table['S8'][1]:>9f}  |  {nickel_table['S8'][2]:>9.3%} |" )
    print( f"|{planes[9]:>6s} | {refl[9]:>8.5f} | {nickel_table['A9'][0]:>8.5f} ± {nickel_table['A9'][1]:>9f}  |  {nickel_table['A9'][2]:>8.3%} ||\
          {nickel_table['S9'][0]:>8.5f} ± {nickel_table['S9'][1]:>9f}  |  {nickel_table['S9'][2]:>9.3%} |" )
    print(103*'-')


    plt.figure(figsize=(9,6))
    plt.plot(nickel.x,model, 'r-',label='Fit')
    plt.plot(nickel.x,nickel.y, 'b+',label='Ni powder data')
    plt.plot(nickel.x,nickel.y-model, 'g-+',label='Residuals')

    plt.legend(loc='best')
    plt.title('Final fit for '+nickel.filename+', '+d4.getDate())
    plt.xlabel(r'$2\theta$'+' (˚)')
    plt.ylabel('Intensity (arb. units)')

    ite =result.fit_report().find('function evals')
    chi = result.fit_report().find('chi-square')
    red = result.fit_report().find('reduced chi-square')
    aka = result.fit_report().find('Akaike')
    ite = result.fit_report()[ite+18:chi-62]
    chi = result.fit_report()[chi+20:red-5]
    red = result.fit_report()[red+20:aka-5]
    info_fit = "Iterations ={}\nchi-sq ={}\nreduced chi-sq ={}\nNumors: {}".format(ite,chi,red,nickel.head[6][15:28])
#    info_fit = "Iterations ={}\nchi-sq ={}\nreduced chi-sq ={}".format(ite,chi,red)

    signature = 'd4nifit.ipynb (Dec 2022), G.J. Cuello (ILL), cuello@ill.fr'

    plt.text(xy_fit[0],xy_fit[1],
             s=info_fit, bbox=dict(facecolor='white', alpha=1.0, edgecolor='black'))
    plt.text(xy_res[0],xy_res[1],
             s=wlength+'\n'+zac, bbox=dict(facecolor='white', alpha=1.0, edgecolor='black'))
    plt.text (axes[0]-0.05*(axes[1]-axes[0]),axes[2]-0.1*(axes[3]-axes[2]),s=signature, fontsize=6)

    planes_fit = ''
    for i in range(10):
        planes_fit += '{}: {:.1f}˚\n'.format(planes[i], refl[i])
    plt.text(xy_ref[0],xy_ref[1],
             s=planes_fit[:-1], bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    plt.axis(axes)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(nickel.basename+'.png')
    print('Results saved as '+nickel.basename+'.png')
    return





###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#
#  Element
#
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

class elemento:
    '''
    class elemento:


        Class that saves by default every nuclear value given by the "Bound Scattering Lengths and Cross Sections of the Elements and Their      Isotopes" in Appendix 2 of Neutron Scattering Fundamentals, volume 44 (dictionary at the bottom called Isotope_dict) inside an object.

    The user can also replace any default value with one of his choice or use this class to consult the value of any isotope variable.

    object = elemento( Iso_name, table, changes)

    Input:
         - Iso_name : (string) Name of the isotope, written as 'Li', 'Li6', 'Li7'... If the isotope does not exist, a warning will be                printed

         - Table : (int) This is a optional input with a default value of 0, which corresponds with no display. Type 1 if the user wants              to have a table with all the values linked to the isotope used displayed on screen. Type 2 if you want all the tables from all            the isotope of the same element displayed.

         - Changes : (string = float) This is an optional input, you can write as many as you want. If you want to use a different value              than the ones from the Isotope_dict dictionary, type the name of the variable, = , the new value. If this new value is somehow            incoherent with its physical meaning, the user will recieve a warning as a print.

   Titanium = elemento('Ti')         Creates an object containing the default values of natural titanium. No output
   Titanium = elemento('Ti', 1)      Creates an object containing the default values of natural titanium and displays them on screen
   Titanium = elemento('Ti', 2)      Creates an object containing the default values of natural titanium and displays one tables of                                            Titanium isotopes on screen

   Titanium = elemento('Ti', 1, A=5, weight = 3)   Creates an object containing the default values of natural titanium except for the                                                        atomic number A that is assigned the value 5, and the weight, replaced by 3. No output

   The keywords used are:

        symbol : (str) symbol of the isotope
        iso: (str) isotope (‘nat’, 1,2,3…)
        name: (str) name of the element
        A : (int)  mass number/ number of neutrons and protons
        Z : (int) Atomic number/ number of protons
        weight : (float) number of protons and neutrons times their masses
        excess : (float)
        spin : (str) nuclear spin
        parity : (str) nuclear parity
        abundance: (float) nuclear abundance (%) for a stable nuclei
        life: (str) half-life for a unstable nuclei followed by its units
        re_bcoh: (float) real part of the bound coherent scattering length
        im_bcoh: (float) imaginary part of the bound coherent scattering length
        bplus: (float) bound scattering length for the (I + ½) state
        bminus: (float) bound scattering length for the (I - ½) state
        re_binc: : (float) real part of the bound incoherent scattering length
        im_binc: (float) imaginary part of the bound incoherent scattering length
        sig_coh: (float) module of bound coherent cross section (σ)
        sig_inc: (float)  module of bound incoherent cross section
        sig_sca: (float) total bound scattering cross section
        sig_abs: (float) absorption cross section for thermal (ν=2200 m/s) neutrons
        contrast:
        neut: (int) number of neutrons, A-Z
        bcoh: (complex) total coherent scattering length
        binc: (complex) total incoherent scattering length
        scoh_bound: (complex) bound coherent cross section

                𝑠𝑐𝑜ℎ_𝑏𝑜𝑢𝑛𝑑=4𝜋 𝑏𝑐𝑜ℎ∙𝑏𝑐𝑜ℎ'        bcoh' = conjugate

        sinc_bound: (complex) bound incoherent cross section

                𝑠𝑖𝑛𝑐_𝑏𝑜𝑢𝑛𝑑=4𝜋 𝑏𝑖𝑛𝑐∙𝑏𝑖𝑛𝑐'         binc' = conjugate
    '''

    def __init__(self, Iso_name , table =0, **kwargs):
        if ((Iso_name in isotope_dict) == False):
            print("\n The isotope you wrote does not exist in this dictionary")

            # dicSearch = dict(list(zip(nombres, isotope_dict[Iso_name[0]])))# take parameters from table
            # if (table ==1):
            #     # Print the names of the columns.
            #     print("{:<15}".format(Iso_name))
            #     print("{:<10} {:<10} {:<10}".format('PARAMETER', 'VALUE', 'UNITS'))
            #     # print each data item.
            #     for key in dicSearch:
            #         print("{:<10} {:<10} {:<10}".format(key, dicSearch[key], units[key]))
            # if (table==2):
            #     for key in isotope_dict:
            #         if isotope_dict[key][0] == isotope_dict[Iso_name][0]:
            #             doc = dict(list(zip(nombres, isotope_dict[key])))
            #             print("\n")
            #             print("{:<15}".format(key.upper()))
            #             print("\n")
            #             print("{:<10} {:<10} {:<10}".format('PARAMETER', 'VALUE', 'UNITS'))
            #             for key in doc:
            #                 print("{:<10} {:<10} {:<10}".format(key, dicSearch[key], units[key]))

        else:
            dic = dict(list(zip(nombres, isotope_dict[Iso_name])))# take parameters from table
            for key in dic:
                if dic[key] == 'NULL': # replaces null with zero in all fields
                    dic[key] = 0
            dic['bcoh'] = 0
            dic['binc'] = 0
            dic['scoh_bound'] = 0
            dic['sinc_bound'] = 0
            dic['neut']= 0
            for key, value in kwargs.items():
                if (key != 'b_length' and key!= 's_bound' and key!= 's_free'):
                    # print("\n")
                    for element in myCorr[key]:
                        # print("\n")
                        element(key, float(value))
                    # print("\n")
                    dic[key] = value

            bcoh = complex(float(dic['re_bcoh']), float(dic['im_bcoh']))
            binc = complex(float(dic['re_binc']), float(dic['im_binc']))
            dic['bcoh'] = bcoh
            dic['binc'] = binc

            scoh_bound = 4 * np.pi * bcoh * bcoh.conjugate()
            sinc_bound = 4 * np.pi * binc * binc.conjugate()

            dic['scoh_bound'] = complex(round(scoh_bound.real, 4),round(scoh_bound.imag, 4))
            dic['sinc_bound'] = complex(round(sinc_bound.real, 4), round(sinc_bound.imag, 4))
            dic['neut'] = dic['A']-dic['Z']

            for key, value in kwargs.items():
                if (key == ('b_length' or 's_bound' or 's_free')) :
                    for element in myCorr[key]:
                        element(key, float(value))
                    dic[key]= value
            if (table ==1):
                # Print the names of the columns.
                print("{:<15}".format(Iso_name))
                print("{:<10} {:<10} {:<10}".format('PARAMETER', 'VALUE', 'UNITS'))
                # print each data item.
                for key in dic:
                    print("{:<10} {:<10} {:<10}".format(key, dic[key], units[key]))
            if (table==2):
                for key in isotope_dict:
                    if isotope_dict[key][0] == isotope_dict[Iso_name][0]:
                        doc = dict(list(zip(nombres, isotope_dict[key])))
                        print("\n")
                        print("{:<15}".format(key.upper()))
                        print("\n")
                        print("{:<10} {:<10} {:<10}".format('PARAMETER', 'VALUE', 'UNITS'))
                        for key in doc:
                            print("{:<10} {:<10} {:<10}".format(key, dic[key], units[key]))

            self.symbol = dic['symbol']
            self.iso = dic['iso']
            self.name = dic['name']
            self.A = dic['A']
            self.Z = dic['Z']
            self.weight = dic['weight']
            self.excess = dic['excess']
            self.spin = dic['spin']
            self.parity = dic['parity']
            self.abundance = dic['abundance']
            self.life = dic['life']
            self.re_bcoh = dic['re_bcoh']
            self.im_bcoh = dic['im_bcoh']
            self.bplus = dic['bplus']
            self.bminus = dic['bminus']
            self.re_binc = dic['re_binc']
            self.im_binc = dic['im_binc']
            self.sig_coh = dic['sig_coh']
            self.sig_inc = dic['sig_inc']
            self.sig_sca = dic['sig_sca']
            self.sig_abs = dic['sig_abs']
            self.contrast = dic['contrast']
            self.neut = dic['neut']
            self.scoh_bound = dic['scoh_bound']
            self.sinc_bound = dic['sinc_bound']
            self.bcoh = dic['bcoh']
            self.binc = dic['binc']

    @classmethod
    def getIsotope(self, Iso_name):
        lista=[]

        # print(name)
        if Iso_name=="All":
            xName=""
            for key in isotope_dict:
                if isotope_dict[key][2] != xName:
                    lista.append(isotope_dict[key][2])
                xName=isotope_dict[key][2]
        else:
            for key in isotope_dict:
                if isotope_dict[key][2] == Iso_name:
                    if(isotope_dict[key][1]=="nat"):
                        lista.append(isotope_dict[key][0])
                    else:
                        lista.append(isotope_dict[key][0]+isotope_dict[key][1])
        return lista

def Positive(key, number):
    if((float(number) > 0) == False):
        print(" WARNING. ", key, " parameter must have a positive value")
def Negative(key, number):
    if ((float(number) < 0) == False):
        print(" WARNING.", key, "parameter must have a negative value")
def SemiWhole(key, number):
    if((float(number)%(1/2)) != 0):
        print(" WARNING.", key, "parameter must be semi-whole")
def Top(key, value):
    if (key == 'weight'): top = 248.072
    if (key == 'abundance'): top = 100
    if (key == 'Z'): top = 96
    if(value > top):
        print(" WARNING.", key, "parameter can't be bigger than", top)
def Int(key, value):
    if (value.is_integer() == False):
        print(" WARNING.", key, "parameter has to be a whole number")
def String(key, value):
    if (isinstance(value, str) == False):
        print(" WARNING.", key, "parameter has to be a string")
def Complex(key, value):
    if (isinstance(value, complex) == False):
        print(" WARNING.", key, "parameter has to be a complex number")
def getFree(isot,key):
    s = complex(round(math.pow((isot.A/(isot.A +1)), 2) * key.real, 4), round(math.pow((isot.A/(isot.A +1)), 2) * key.imag, 4))
    return s
def dummyOk(key, value):
    return

myCorr = {
    "symbol": [String],
    "iso": [String],
    "name": [String],
    "A": [Int, Positive],
    "Z": [Int,Positive],
    "weight": [Top, Positive],
    #"excess": ,
    "spin": [SemiWhole],
    #"parity": ,
    "abundance": [Top, Positive],
    "life": [Positive, String],
    "re_bcoh": [dummyOk],
    #"im_bcoh": ,
    #"bplus": ,
    #"bminus": ,
    #"re_binc": ,
    #"im_binc": ,
    #"sig_coh": ,
    #"sig_inc": ,
    #"sig_sca": ,
    #"sig_abs": ,
    #"contrast": ,
    "neut": [Int, Positive],
    "bcoh": [Complex],
    "binc": [Complex],
    "scoh_bound": [Complex] ,
    "sinc_bound":[Complex]
}
units = {
    "symbol":'-'  ,
    "iso": '-',
    "name":'-' ,
    "A": '-',
    "Z": '-',
    "weight": 'uma',
    "excess":'-' ,
    "spin":'-',
    "parity":'-' ,
    "abundance": '%',
    "life":'-' ,
    "re_bcoh": 'fm',
    "im_bcoh":'fm' ,
    "bplus": 'fm',
    "bminus": 'fm',
    "re_binc": 'fm',
    "im_binc": 'fm',
    "sig_coh": 'barns',
    "sig_inc": 'barns',
    "sig_sca": 'barns',
    "sig_abs": 'barns',
    "contrast":'-' ,
    "neut":'-' ,
    "bcoh":'fm' ,
    "binc":'fm',
    "scoh_bound":'barns',
    "sinc_bound":'barns'
}

nombres = ["symbol", "iso", "name", "A", "Z", "weight", "excess", "spin", "parity", "abundance", "life", "re_bcoh","im_bcoh","bplus", "bminus", "re_binc", "im_binc", "sig_coh", "sig_inc", "sig_sca", "sig_abs", "contrast", "neut", "bcoh", "binc", "scoh_bound", "sinc_bound"]
isotope_dict = {
'n1': ('n', '1', 'neutron', 1, 0, 1.00866, 8.07132, '1/2', '+', 'NULL', '10.24m', -37.8, 0, 'NULL', 'NULL', 'NULL', 'NULL', 44.89, 0, 44.89, 0, 'NULL'),
'H1': ('H', '1', 'proton', 1, 1, 1.00728, 6.77799, '1/2', '+', 99.985, 'inf', -37.7423, 0, 10.817, -47.42, 25.217, 0, 1.7589, 79.91, 81.6689, 0.3326, 100.893),
'H2': ('H', '2', 'deuteron', 2, 1, 2.01355, 12.6247, '1', '+', 0.015, 'inf', 6.674, 0, 9.53, 0.975, 4.03, 0, 5.597, 2.04, 7.637, 0.000519, 2.18612),
'H3': ('H', '3', 'triton', 3, 1, 3.0155, 14.4388, '1/2', '+', 0, '12.312y', 4.792, 0, 4.18, 6.56, -1.04, 0, 2.89, 0.14, 3.03, 0.000006, 0.642565),
'H': ('H', 'nat', 'Hydrogen', 1, 1, 1.00794, 0, 'NULL', 'NULL', 100, 'inf', -3.739, 0, 'NULL', 'NULL', 'NULL', 'NULL', 1.7568, 80.26, 82.0168, 0.3326, 0),
'He3': ('He', '3', 'Helium', 3, 2, 'NULL', 14.9312, '1/2', '+', 0.000137, 'inf', 5.74, 1.483, 4.5, 9.3, -2.1, 2.568, 4.42, 1.38, 5.8, 5333, 2.30713),
'He4': ('He', '4', 'Helium', 4, 2, 'NULL', 2.4249, '0', '+', 99.9999, 'inf', 3.26, 0, 'NULL', 'NULL', 0, 0, 1.34, 0, 1.34, 0, 0),
'He': ('He', 'nat', 'Helium', 4, 2, 4.0026, 0, 'NULL', 'NULL', 100, 'inf', 3.26, 0, 'NULL', 'NULL', 'NULL', 'NULL', 1.34, 0, 1.34, 0.00747, 0),
'Li6': ('Li', '6', 'Lithium', 6, 3, 'NULL', 14.0868, '1', '+', 7.59, 'inf', 2, 0.261, 0.67, 4.67, -1.89, 0.26, 0.51, 0.46, 0.97, 940, 0.126903),
'Li7': ('Li', '7', 'Lithium', 7, 3, 'NULL', 14.9081, '3/2', '-', 92.41, 'inf', -2.22, 0, -4.15, 1, -2.49, 0, 0.619, 0.78, 1.399, 0.0454, 0.365208),
'Li': ('Li', 'nat', 'Lithium', 7, 3, 6.941, 0, 'NULL', 'NULL', 100, 'inf', -1.9, 0, 'NULL', 'NULL', 'NULL', 'NULL', 0.454, 0.92, 1.374, 70.5, 0),
'Be': ('Be', 'nat', 'Beryllium', 9, 4, 9.01218, 11.3476, '3/2', '-', 100, 'inf', 7.79, 0, 'NULL', 'NULL', 0.12, 0, 7.63, 0.0018, 7.6318, 0.0076, 0),
'B10': ('B', '10', 'Boron', 10, 5, 'NULL', 12.0507, '3', '+', 19.8, 'inf', 0.1, -1.066, -4.2, 5.2, -4.7, 1.231, 0.144, 3, 3.144, 3835, 0.959256),
'B11': ('B', '11', 'Boron', 11, 5, 'NULL', 8.6679, '3/2', '-', 80.2, 'inf', 6.65, 0, 5.6, 8.3, -1.3, 0, 5.56, 0.21, 5.77, 20.0055, 0.571776),
'B': ('B', 'nat', 'Boron', 11, 5, 10.811, 0, 'NULL', 'NULL', 100, 'inf', 5.3, -0.213, 'NULL', 'NULL', 'NULL', 'NULL', 3.54, 1.7, 5.24, 767, 0),
'C12': ('C', '12', 'Carbon', 12, 6, 'NULL', 0, '0', '+', 98.89, 'inf', 6.6535, 0, 'NULL', 'NULL', 0, 0, 5.559, 0, 5.559, 0.00353, 0.00153479),
'C13': ('C', '13', 'Carbon', 13, 6, 'NULL', 3.125, '1/2', '-', 1.11, 'inf', 6.19, 0, 5.6, 6.2, -0.25, 0, 4.81, 0.034, 4.844, 0.00137, 0.133144),
'C': ('C', 'nat', 'Carbon', 12, 6, 12.0107, 0, 'NULL', 'NULL', 100, 'inf', 6.6484, 0, 'NULL', 'NULL', 'NULL', 'NULL', 5.551, 0.001, 5.552, 0.0035, 0),
'N14': ('N', '14', 'Nitrogen', 14, 7, 'NULL', 2.8634, '1', '+', 99.634, 'inf', 9.37, 0, 10.7, 6.2, 2.1, 0, 11.03, 0.5, 11.53, 1.91, 0.00213789),
'N15': ('N', '15', 'Nitrogen', 15, 7, 'NULL', 0.1014, '1/2', '-', 0.366, 'inf', 6.44, 0, 6.77, 6.21, 0.24, 0, 5.21, 0.00005, 5.21005, 0.000024, 0.526609),
'N': ('N', 'nat', 'Nitrogen', 14, 7, 14.0067, 0, 'NULL', 'NULL', 100, 'inf', 9.36, 0, 'NULL', 'NULL', 'NULL', 'NULL', 11.01, 0.5, 11.51, 1.9, 0),
'O16': ('O', '16', 'Oxygen', 16, 8, 'NULL', -4.737, '0', '+', 99.762, 'inf', 5.805, 0, 'NULL', 'NULL', 0, 0, 4.232, 0, 4.232, 0.0001, 0),
'O17': ('O', '17', 'Oxygen', 17, 8, 'NULL', -0.8088, '5/2', '+', 0.038, 'inf', 5.66, 0, 5.86, 5.41, 0.17, 0, 4.2, 0.004, 4.204, 0.236, 0.049333),
'O18': ('O', '18', 'Oxygen', 18, 8, 'NULL', -0.7815, '0', '+', 0.2, 'inf', 5.84, 0, 'NULL', 'NULL', 0, 0, 4.29, 0, 4.29, 0.00016, 0.0120949),
'O': ('O', 'nat', 'Oxygen', 16, 8, 15.9994, 0, 'NULL', 'NULL', 100, 'inf', 5.805, 0, 'NULL', 'NULL', 'NULL', 'NULL', 4.232, 0, 4.232, 0.00019, 0),
'F': ('F', 'nat', 'Fluorine', 19, 9, 18.9984, -1.4874, '1/2', '+', 100, 'inf', 5.654, 0, 5.632, 5.767, -0.082, 0, 4.017, 0.0008, 4.0178, 0.0096, 0),
'Ne20': ('Ne', '20', 'Neon', 20, 10, 'NULL', -7.0419, '0', '+', 90.48, 'inf', 4.631, 0, 'NULL', 'NULL', 0, 0, 2.695, 0, 2.695, 0.036, 0.028674),
'Ne21': ('Ne', '21', 'Neon', 21, 10, 'NULL', -5.7318, '3/2', '+', 0.27, 'inf', 6.66, 0, 'NULL', 'NULL', 0.6, 0, 5.6, 0.05, 5.65, 0.67, 1.12753),
'Ne22': ('Ne', '22', 'Neon', 22, 10, 'NULL', -8.0247, '0', '+', 9.25, 'inf', 3.87, 0, 'NULL', 'NULL', 0, 0, 1.88, 0, 1.88, 0.046, 0.281627),
'Ne': ('Ne', 'nat', 'Neon', 20, 10, 20.1797, 0, 'NULL', 'NULL', 100, 'inf', 4.566, 0, 'NULL', 'NULL', 'NULL', 'NULL', 2.62, 0.008, 2.628, 0.039, 0),
'Na': ('Na', 'nat', 'Sodium', 23, 11, 22.9898, -9.5299, '3/2', '+', 100, 'inf', 3.63, 0, 6.42, -1, 3.59, 0, 1.66, 1.62, 3.28, 0.53, 0),
'Mg24': ('Mg', '24', 'Magnesium', 24, 12, 'NULL', -13.9336, '0', '+', 78.99, 'inf', 5.49, 0, 'NULL', 'NULL', 0, 0, 4.03, 0, 4.03, 0.05, 0.0432485),
'Mg25': ('Mg', '25', 'Magnesium', 25, 12, 'NULL', -13.1928, '5/2', '+', 10, 'inf', 3.62, 0, 4.73, 1.76, 1.48, 0, 1.65, 0.28, 1.93, 0.19, 0.546413),
'Mg26': ('Mg', '26', 'Magnesium', 26, 12, 'NULL', -16.2146, '0', '+', 11.01, 'inf', 4.89, 0, 'NULL', 'NULL', 0, 0, 3, 0, 3, 0.0382, 0.172323),
'Mg': ('Mg', 'nat', 'Magnesium', 24, 12, 24.305, 0, 'NULL', 'NULL', 100, 'inf', 5.375, 0, 'NULL', 'NULL', 'NULL', 'NULL', 3.631, 0.08, 3.711, 0.063, 0),
'Al': ('Al', 'nat', 'Aluminium', 27, 13, 26.9815, -17.1967, '5/2', '+', 100, 'inf', 3.449, 0, 3.67, 3.15, 0.256, 0, 1.495, 0.0082, 1.5032, 0.231, 0),
'Si28': ('Si', '28', 'Silicon', 28, 14, 27.9769, -21.4928, '0', '+', 92.22, 'inf', 4.106, 0, 'NULL', 'NULL', 0, 0, 2.12, 0, 2.12, 0.177, 0.0214273),
'Si29': ('Si', '29', 'Silicon', 29, 14, 28.9765, -21.895, '1/2', '+', 4.69, 'inf', 4.7, 0, 4.5, 4.7, -1.08, 0, 2.78, 0.79, 3.57, 0.101, 0.282186),
'Si30': ('Si', '30', 'Silicon', 30, 14, 29.9738, -24.4329, '0', '+', 3.09, 'inf', 4.58, 0, 'NULL', 'NULL', 0, 0, 2.64, 0, 2.64, 0.107, 0.217548),
'Si': ('Si', 'nat', 'Silicon', 28, 14, 28.085, 'NULL', 'NULL', 'NULL', 100, 'inf', 4.15071, 0, 'NULL', 'NULL', 'NULL', 'NULL', 2.1633, 0.004, 2.167, 0.171, 0),
'P': ('P', 'nat', 'Phosphorus', 31, 15, 30.974, -24.4409, '1/2', '+', 100, 'inf', 5.13, 0, 'NULL', 'NULL', 0.3, 0, 3.307, 0.005, 3.312, 0.172, 0),
'S32': ('S', '32', 'Sulfur', 32, 16, 31.9721, -26.0157, '0', '+', 95.02, 'inf', 2.804, 0, 'NULL', 'NULL', 0, 0, 0.988, 0, 0.988, 0.54, 0.0299791),
'S33': ('S', '33', 'Sulfur', 33, 16, 32.9715, -26.586, '3/2', '+', 0.75, 'inf', 4.74, 0, 'NULL', 'NULL', 1.5, 0, 2.8, 0.3, 3.1, 0.54, 1.77193),
'S34': ('S', '34', 'Sulfur', 34, 16, 33.9679, -29.9318, '0', '+', 4.21, 'inf', 3.48, 0, 'NULL', 'NULL', 0, 0, 1.52, 0, 1.52, 0.227, 0.494113),
'S36': ('S', '36', 'Sulfur', 36, 16, 35.9671, -28.8464, '0', '+', 0.02, 'inf', 3, 0, 'NULL', 'NULL', 0, 0, 1.1, 0, 1.1, 0.15, 0.11037),
'S': ('S', 'nat', 'Sulfur', 32, 16, 32.065, 0, 'NULL', 'NULL', 100, 'inf', 2.847, 0, 'NULL', 'NULL', 'NULL', 'NULL', 1.0186, 0.007, 1.0256, 0.53, 0),
'Cl35': ('Cl', '35', 'Chlorine', 35, 17, 34.9689, -29.0135, '3/2', '+', 75.78, 'inf', 11.709, 0, 16.3, 4, 0, 0, 17.06, 4.7, 21.8, 44.1, 0.494105),
'Cl37': ('Cl', '37', 'Chlorine', 37, 17, 36.9659, -31.7615, '3/2', '+', 24.22, 'inf', 3.08, 0, 3.1, 3.05, 0.02, 0, 1.19, 0.001, 1.19, 0.433, 0.896618),
'Cl': ('Cl', 'nat', 'Chlorine', 35, 17, 35.453, 0, 'NULL', 'NULL', 100, 'inf', 9.5792, 0, 'NULL', 'NULL', 'NULL', 'NULL', 11.528, 5.3, 16.828, 33.5, 0),
'Ar36': ('Ar', '36', 'Argon', 36, 18, 35.6975, -30.2315, '0', '+', 0.337, 'inf', 24.9, 0, 'NULL', 'NULL', 0, 0, 77.9, 0, 77.9, 5.2, 169.132),
'Ar38': ('Ar', '38', 'Argon', 38, 18, 37.9627, -34.7146, '0', '+', 0.063, 'inf', 3.5, 0, 'NULL', 'NULL', 0, 0, 1.5, 0, 1.5, 0.8, 2.36143),
'Ar40': ('Ar', '40', 'Argon', 40, 18, 39.9624, -35.0399, '0', '+', 99.6, 'inf', 1.84, 0, 'NULL', 'NULL', 0, 0, 0.421, 0, 0.421, 0.66, 0.0709827),
'Ar': ('Ar', 'nat', 'Argon', 40, 18, 39.948, 'NULL', '0', '+', 100, 'inf', 1.909, 0, 'NULL', 'NULL', 'NULL', 'NULL', 0.458, 0.225, 0.683, 0.675, 0),
'K39': ('K', '39', 'Potassium', 39, 19, 38.9637, -33.807, '3/2', '+', 93.2581, 'inf', 3.72, 0, 5.15, 5.15, 1.43, 0, 1.76, 0.25, 2.01, 2.1, 0.0274336),
'K40': ('K', '40', 'Potassium', 40, 19, 39.964, -33.5352, '4', '-', 0.0117, 'inf', 3.1, 0, 'NULL', 'NULL', 'NULL', 'NULL', 1.1, 0.5, 1.6, 35, 0.286504),
'K41': ('K', '41', 'Potassium', 41, 19, 40.9618, -35.5591, '3/2', '+', 6.7302, 'inf', 2.69, 0, 'NULL', 'NULL', 1.51, 0, 0.91, 0.3, 1.2, 1.46, 0.462755),
'K': ('K', 'nat', 'Potassium', 39, 19, 39.0983, 'NULL', 'NULL', 'NULL', 100, 'inf', 3.67, 0, 'NULL', 'NULL', 'NULL', 'NULL', 1.69, 0.27, 1.96, 2.1, 0),
'Ca40': ('Ca', '40', 'Calcium', 40, 20, 39.9626, -34.8463, '0', '+', 96.941, 'inf', 4.78, 0, 'NULL', 'NULL', 0, 0, 2.9, 0, 2.9, 0.41, 0.0343323),
'Ca42': ('Ca', '42', 'Calcium', 42, 20, 41.9586, -38.5471, '0', '+', 0.647, 'inf', 3.36, 0, 'NULL', 'NULL', 0, 0, 1.42, 0, 1.42, 0.68, 0.488927),
'Ca43': ('Ca', '43', 'Calcium', 43, 20, 42.9588, -38.4086, '7/2', '-', 0.135, 'inf', -1.56, 0, 'NULL', 'NULL', 'NULL', 'NULL', 0.31, 0.5, 0.8, 6.2, 0.889832),
'Ca44': ('Ca', '44', 'Calcium', 44, 20, 43.9555, -41.4685, '0', '+', 2.086, 'inf', 1.42, 0, 'NULL', 'NULL', 0, 0, 0.25, 0, 0.25, 0.88, 0.908719),
'Ca46': ('Ca', '46', 'Calcium', 46, 20, 45.9537, -43.1351, '0', '+', 0.004, 'inf', 3.55, 0, 'NULL', 'NULL', 0, 0, 1.6, 0, 1.6, 0.7, 0.429493),
'Ca48': ('Ca', '48', 'Calcium', 48, 20, 47.9525, -44.2141, '0', '+', 0.187, 'inf', 0.39, 0, 'NULL', 'NULL', 0, 0, 0.019, 0, 0.019, 1.09, 0.993115),
'Ca': ('Ca', 'nat', 'Calcium', 40, 20, 40.078, 'NULL', 'NULL', 'NULL', 100, 'inf', 4.7, 0, 'NULL', 'NULL', 'NULL', 'NULL', 2.78, 0.05, 2.83, 0.43, 0),
'Sc': ('Sc', 'nat', 'Scandium', 35, 21, 44.9559, -41.0678, '7/2', '-', 100, 'inf', 12.1, 0, 6.91, 18.99, -6.02, 0, 19.03, 4.5, 23.5, 27.5, 0),
'Ti46': ('Ti', '46', 'Titanium', 46, 22, 'NULL', -44.1234, '0', '+', 8.25, 'inf', 4.72, 0, 'NULL', 'NULL', 0, 0, 3.05, 0, 3.05, 0.59, 0.961662),
'Ti47': ('Ti', '47', 'Titanium', 47, 22, 'NULL', -44.9324, '5/2', '-', 7.44, 'inf', 3.53, 0, 0.46, 7.64, -3.5, 0, 1.66, 1.5, 3.16, 1.7, 0.0972096),
'Ti48': ('Ti', '48', 'Titanium', 48, 22, 'NULL', -48.4877, '0', '+', 73.72, 'inf', -5.86, 0, 'NULL', 'NULL', 0, 0, 4.65, 0, 4.65, 7.84, 2.02368),
'Ti49': ('Ti', '49', 'Titanium', 49, 22, 'NULL', -48.5588, '7/2', '-', 5.41, 'inf', 0.98, 0, 2.6, -1.2, 1.9, 0, 0.14, 3.3, 3.44, 2.2, 0.915435),
'Ti50': ('Ti', '50', 'Titanium', 50, 22, 'NULL', -51.4267, '0', '+', 5.18, 'inf', 5.88, 0, 'NULL', 'NULL', 0, 0, 4.8, 0, 4.8, 0.179, 2.04435),
'Ti': ('Ti', 'nat', 'Titanium', 48, 22, 47.867, 0, 'NULL', 'NULL', 100, 'inf', -3.37, 0, 'NULL', 'NULL', 'NULL', 'NULL', 1.485, 2.87, 4.355, 6.09, 0),
'V50': ('V', '50', 'Vanadium', 50, 23, 'NULL', -49.2216, '6', '+', 0.25, 'inf', 7.6, 0, 'NULL', 'NULL', 'NULL', 'NULL', 7.3, 0.5, 7.8, 60, 293.32),
'V51': ('V', '51', 'Vanadium', 51, 23, 'NULL', -52.2014, '7/2', '-', 99.75, 'inf', -0.402, 0, 4.93, -7.58, 6.35, 0, 0.0203, 5.07, 5.0903, 4.9, 0.176536),
'V': ('V', 'nat', 'Vanadium', 51, 23, 50.9415, 0, 'NULL', 'NULL', 100, 'inf', -0.443, 0, 'NULL', 'NULL', 'NULL', 'NULL', 0.01838, 5.08, 5.09838, 5.08, 0),
'Cr50': ('Cr', '50', 'Chromium', 50, 24, 49.946, -50.2595, '0', '+', 4.345, 'inf', -4.5, 0, 'NULL', 'NULL', 0, 0, 2.54, 0, 2.54, 15.8, 0.532555),
'Cr52': ('Cr', '52', 'Chromium', 52, 24, 51.9405, -55.4169, '0', '+', 83.789, 'inf', 4.914, 0, 'NULL', 'NULL', 0, 0, 3.042, 0, 3.042, 0.76, 0.827517),
'Cr53': ('Cr', '53', 'Chromium', 53, 24, 52.9407, -55.2847, '3/2', '-', 9.501, 'inf', -4.2, 0, 1.16, -13, 6.9, 0, 2.22, 5.93, 8.15, 18.1, 0.335026),
'Cr54': ('Cr', '54', 'Chromium', 54, 24, 53.9389, -56.9325, '0', '+', 2.365, 'inf', 4.55, 0, 'NULL', 'NULL', 0, 0, 2.6, 0, 2.6, 0.36, 0.566801),
'Cr': ('Cr', 'nat', 'Chromium', 52, 24, 51.9961, 'NULL', 'NULL', 'NULL', 100, 'inf', 3.635, 0, 'NULL', 'NULL', 'NULL', 'NULL', 1.66, 1.83, 3.49, 3.05, 0),
'Mn': ('Mn', 'nat', 'Manganese', 55, 25, 54.938, -57.7106, '5/2', '-', 100, 'inf', -3.75, 0, -4.93, -1.46, -1.71, 0, 1.75, 0.4, 2.15, 13.3, 0),
'Fe54': ('Fe', '54', 'Iron', 54, 26, 53.9396, -56.2525, '0', '+', 5.845, 'inf', 4.2, 0, 'NULL', 'NULL', 0, 0, 2.2, 0, 2.2, 2.25, 0.802469),
'Fe56': ('Fe', '56', 'Iron', 56, 26, 55.9349, -60.6054, '0', '+', 91.754, 'inf', 10.1, 0, 'NULL', 'NULL', 0, 0, 12.42, 0, 12.42, 2.59, 0.142297),
'Fe57': ('Fe', '57', 'Iron', 57, 26, 56.9354, -60.1801, '1/2', '-', 2.119, 'inf', 2.3, 0, 'NULL', 'NULL', 'NULL', 'NULL', 0.66, 0.3, 1.03, 2.48, 0.940763),
'Fe58': ('Fe', '58', 'Iron', 58, 26, 57.9333, -62.1534, '0', '+', 0.282, 'inf', 15, 0, 'NULL', 'NULL', 0, 0, 28, 0, 28, 1.28, 1.51953),
'Fe': ('Fe', 'nat', 'Iron', 56, 26, 55.845, 'NULL', 'NULL', 'NULL', 100, 'inf', 9.45, 0, 'NULL', 'NULL', 'NULL', 'NULL', 11.22, 0.4, 11.62, 2.56, 0),
'Co': ('Co', 'nat', 'Cobalt', 59, 27, 58.9332, -62.2284, 'NULL', 'NULL', 100, 'inf', 2.49, 0, -9.21, 3.58, -6.43, 0, 0.779, 4.8, 5.6, 37.18, 0),
'Ni58': ('Ni', '58', 'Nickel', 58, 28, 57.9353, -60.2277, '0', '+', 68.0769, 'inf', 68.077, 0, 'NULL', 'NULL', 0, 0, 26.1, 0, 26.1, 4.6, 42.6844),
'Ni60': ('Ni', '60', 'Nickel', 60, 28, 59.9308, -64.4721, '0', '+', 26.2231, 'inf', 2.8, 0, 'NULL', 'NULL', 0, 0, 0.99, 0, 0.99, 2.9, 0.9261),
'Ni61': ('Ni', '61', 'Nickel', 61, 28, 60.9311, -64.2209, '3/2', '+', 1.1399, 'inf', 7.6, 0, 'NULL', 'NULL', -3.9, 0, 7.26, 1.9, 5.6, 2.5, 0.455557),
'Ni62': ('Ni', '62', 'Nickel', 62, 28, 61.9283, -66.7461, '0', '+', 3.6345, 'inf', -8.7, 0, 'NULL', 'NULL', 0, 0, 9.5, 0, 9.5, 14.5, 0.286549),
'Ni64': ('Ni', '64', 'Nickel', 64, 28, 63.928, -67.0993, '0', '+', 0.9256, 'inf', -0.37, 0, 'NULL', 'NULL', 0, 0, 0.017, 0, 0.017, 1.52, 0.99871),
'Ni': ('Ni', 'nat', 'Nickel', 59, 28, 58.6934, 'NULL', 'NULL', 'NULL', 100, 'inf', 10.3, 0, 'NULL', 'NULL', 'NULL', 'NULL', 13.3, 5.2, 18.5, 4.49, 0),
'Cu63': ('Cu', '63', 'Copper', 63, 29, 'NULL', -65.5795, '3/2', '-', 69.15, 'inf', 6.477, 0, 'NULL', 'NULL', 0.22, 0, 5.2, 0.006, 5.206, 4.5, 0.295732),
'Cu65': ('Cu', '65', 'Copper', 65, 29, 'NULL', -67.2637, '3/2', '-', 30.85, 'inf', 10.204, 0, 'NULL', 'NULL', 1.82, 0, 14.1, 0.4, 14.5, 2.17, 0.747959),
'Cu': ('Cu', 'nat', 'Copper', 64, 29, 63.546, 0, 'NULL', 'NULL', 100, 'inf', 7.718, 0, 'NULL', 'NULL', 'NULL', 'NULL', 7.485, 0.55, 8.035, 3.78, 0),
'Zn64': ('Zn', '64', 'Zinc', 64, 30, 63.9291, -66.0036, '0', '+', 48.63, 'inf', 5.23, 0, 'NULL', 'NULL', 0, 0, 3.42, 0, 3.42, 0.93, 0.152174),
'Zn66': ('Zn', '66', 'Zinc', 66, 30, 65.926, -68.8994, '0', '+', 27.9, 'inf', 5.98, 0, 'NULL', 'NULL', 0, 0, 4.48, 0, 4.48, 0.62, 0.108423),
'Zn67': ('Zn', '67', 'Zinc', 67, 30, 66.9271, -67.8804, '5/2', '-', 4.1, 'inf', 7.58, 0, 5.8, 10.1, -1.5, 0, 7.18, 0.28, 7.46, 6.8, 0.780909),
'Zn68': ('Zn', '68', 'Zinc', 68, 30, 67.9248, -70.0072, '0', '+', 18.75, 'inf', 6.04, 0, 'NULL', 'NULL', 0, 0, 4.57, 0, 4.57, 1.1, 0.130778),
'Zn70': ('Zn', '70', 'Zinc', 70, 30, 69.9253, -69.5647, '0', '+', 0.62, 'inf', 6, 0, 'NULL', 'NULL', 0, 0, 4.5, 0, 4.5, 0.092, 0.11585),
'Zn': ('Zn', 'nat', 'Zinc', 65, 30, 65.38, 0, 'NULL', 'NULL', 100, 'inf', 5.68, 0, 'NULL', 'NULL', 'NULL', 'NULL', 4.054, 0.077, 4.131, 1.11, 0),
'Ga69': ('Ga', '69', 'Gallium', 69, 31, 68.9256, -69.3278, '3/2', '-', 60.108, 'inf', 8.0403, 0, 6.3, 10.5, -0.85, 0, 7.8, 0.091, 7.89, 2.18, 0.217104),
'Ga71': ('Ga', '71', 'Gallium', 71, 31, 70.9247, -70.1402, '3/2', '-', 39.892, 'inf', 6.17, 0, 5.5, 7.8, -0.82, 0, 5.15, 0.084, 5.23, 3.61, 0.283273),
'Ga': ('Ga', 'nat', 'Gallium', 70, 31, 69.723, 0, 'NULL', 'NULL', 100, 'inf', 7.288, 0, 'NULL', 'NULL', 'NULL', 'NULL', 6.675, 0.16, 6.835, 2.75, 0),
'Ge70': ('Ge', '70', 'Germanium', 70, 32, 69.9242, -70.5631, '0', '+', 20.84, 'inf', 10, 0, 'NULL', 'NULL', 0, 0, 12.6, 0, 12.6, 3, 0.492666),
'Ge72': ('Ge', '72', 'Germanium', 72, 32, 71.9221, -72.5859, '0', '+', 27.54, 'inf', 8.51, 0, 'NULL', 'NULL', 0, 0, 9.1, 0, 9.1, 0.8, 0.0809902),
'Ge73': ('Ge', '73', 'Germanium', 73, 32, 72.9235, -71.2975, '9/2', '+', 7.73, 'inf', 5.02, 0, 5.5, 7.8, 3.43, 0, 3.17, 1.5, 4.7, 15.1, 0.623842),
'Ge74': ('Ge', '74', 'Germanium', 74, 32, 73.9212, -73.4224, '0', '+', 36.28, 'inf', 7.58, 0, 'NULL', 'NULL', 0, 0, 7.2, 0, 7.2, 0.4, 0.142368),
'Ge76': ('Ge', '76', 'Germanium', 76, 32, 75.9214, -73.213, '0', '+', 7.61, 'inf', 8.2, 0, 'NULL', 'NULL', 0, 0, 8, 0, 8, 0.16, 0.0036686),
'Ge': ('Ge', 'nat', 'Germanium', 73, 32, 72.64, 0, 'NULL', 'NULL', 100, 'inf', 8.185, 0, 'NULL', 'NULL', 'NULL', 'NULL', 8.42, 0.18, 8.6, 2.2, 0),
'As': ('As', 'nat', 'Arsenic', 75, 33, 74.9216, -73.0324, '3/2', '-', 100, 'inf', 6.58, 0, 6.04, 7.47, -0.69, 0, 5.44, 0.06, 5.5, 4.5, 0),
'Se74': ('Se', '74', 'Selenium', 74, 34, 73.9225, -72.2127, '0', '+', 0.89, 'inf', 0.8, 0, 'NULL', 'NULL', 0, 0, 0.1, 0, 0.1, 51.8, 0.989925),
'Se76': ('Se', '76', 'Selenium', 76, 34, 75.9192, -75.252, '0', '+', 9.37, 'inf', 12.2, 0, 'NULL', 'NULL', 0, 0, 18.7, 0, 18.7, 85, 1.34317),
'Se77': ('Se', '77', 'Selenium', 77, 34, 76.9199, -74.5996, '1/2', '-', 7.63, 'inf', 8.25, 0, 'NULL', 'NULL', -0.6, 0, 8.6, 0.05, 8.65, 42, 0.0714977),
'Se78': ('Se', '78', 'Selenium', 78, 34, 77.9173, -77.0261, '0', '+', 23.77, 'inf', 8.24, 0, 'NULL', 'NULL', 0, 0, 8.5, 0, 8.5, 0.43, 0.0689017),
'Se80': ('Se', '80', 'Selenium', 80, 34, 79.9165, -77.7599, '0', '+', 49.61, 'inf', 7.48, 0, 'NULL', 'NULL', 0, 0, 7.03, 0, 7.03, 0.61, 0.119181),
'Se82': ('Se', '82', 'Selenium', 82, 34, 81.9167, -77.594, '0', '+', 8.73, 'inf', 6.43, 0, 'NULL', 'NULL', 0, 0, 5.05, 0, 5.05, 0.044, 0.349113),
'Se': ('Se', 'nat', 'Selenium', 79, 34, 78.96, 0, 'NULL', 'NULL', 100, 'inf', 7.97, 0, 'NULL', 'NULL', 'NULL', 'NULL', 7.98, 0.32, 8.3, 11.7, 0),
'Br79': ('Br', '79', 'Bromine', 79, 35, 78.9183, -76.0685, '3/2', '-', 50.69, 'inf', 6.79, 0, 'NULL', 'NULL', -1.1, 0, 5.81, 0.15, 5.96, 11, 0.240261),
'Br81': ('Br', '81', 'Bromine', 81, 35, 80.9163, -77.9748, '3/2', '-', 49.31, 'inf', 6.78, 0, 'NULL', 'NULL', -0.6, 0, 5.79, 0.05, 5.84, 2.7, 0.242497),
'Br': ('Br', 'nat', 'Bromine', 80, 35, 79.904, 'NULL', 'NULL', 'NULL', 100, 'inf', 7.79, 0, 'NULL', 'NULL', 'NULL', 'NULL', 5.8, 0.1, 5.9, 6.9, 0),
'Kr78': ('Kr', '78', 'Krypton', 78, 36, 77.9204, -74.1797, '0', '+', 0.355, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 0, 0, 'NULL', 0, 'NULL', 6.4, 1),
'Kr80': ('Kr', '80', 'Krypton', 80, 36, 79.9164, -77.8925, '0', '+', 2.2286, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 0, 0, 'NULL', 0, 'NULL', 11.8, 1),
'Kr82': ('Kr', '82', 'Krypton', 82, 36, 81.9135, -80.5895, '0', '+', 11.593, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 0, 0, 'NULL', 0, 'NULL', 29, 1),
'Kr83': ('Kr', '83', 'Krypton', 83, 36, 82.9141, -79.9817, '9/2', '+', 11.5, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 0, 'NULL', 185, 1),
'Kr84': ('Kr', '84', 'Krypton', 84, 36, 83.9115, -82.431, '0', '+', 56.987, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 0, 0, 'NULL', 0, 6.6, 0.113, 1),
'Kr86': ('Kr', '86', 'Krypton', 86, 36, 85.9106, -83.2656, '0', '+', 17.279, 'inf', 8.07, 0, 'NULL', 'NULL', 0, 0, 8.2, 0, 8.2, 0.003, 0.0676896),
'Kr': ('Kr', 'nat', 'Krypton', 84, 36, 83.798, 'NULL', 'NULL', 'NULL', 100, 'inf', 7.81, 0, 'NULL', 'NULL', 'NULL', 'NULL', 7.67, 0.01, 7.68, 25, 0),
'Rb85': ('Rb', '85', 'Rubidium', 85, 37, 84.9118, -82.1673, '5/2', '-', 72.17, 'inf', 7.07, 0, 'NULL', 'NULL', 'NULL', 'NULL', 6.2, 0.5, 6.7, 0.48, 0.00282286),
'Rb87': ('Rb', '87', 'Rubidium', 87, 37, 86.9092, -84.5978, '3/2', '-', 27.83, 'inf', 7.27, 0, 'NULL', 'NULL', 'NULL', 'NULL', 6.6, 0.5, 7.1, 0.12, 0.0543925),
'Rb': ('Rb', 'nat', 'Rubidium', 85, 37, 85.4678, 'NULL', 'NULL', 'NULL', 100, 'inf', 7.08, 0, 'NULL', 'NULL', 'NULL', 'NULL', 6.32, 0.5, 6.8, 0.38, 0),
'Sr84': ('Sr', '84', 'Strontium', 84, 38, 83.9134, -80.6438, '0', '+', 0.56, 'inf', 5, 0, 'NULL', 'NULL', 0, 0, 6, 0, 6, 0.87, 0.492699),
'Sr86': ('Sr', '86', 'Strontium', 86, 38, 85.9093, -84.5236, '0', '+', 9.86, 'inf', 5.68, 0, 'NULL', 'NULL', 0, 0, 4.04, 0, 4.04, 1.04, 0.34533),
'Sr87': ('Sr', '87', 'Strontium', 87, 38, 86.9089, -84.8804, '9/2', '+', 7, 'inf', 7.41, 0, 'NULL', 'NULL', 'NULL', 'NULL', 6.88, 0.5, 7.4, 16, 0.114198),
'Sr88': ('Sr', '88', 'Strontium', 88, 38, 87.9056, -87.9217, '0', '+', 82.58, 'inf', 7.16, 0, 'NULL', 'NULL', 0, 0, 6.42, 0, 6.42, 0.058, 0.0402838),
'Sr': ('Sr', 'nat', 'Strontium', 87, 38, 87.62, 'NULL', 'NULL', 'NULL', 100, 'inf', 7.02, 0, 'NULL', 'NULL', 'NULL', 'NULL', 6.19, 0.06, 6.25, 1.28, 0),
'Y': ('Y', 'nat', 'Yttrium', 89, 39, 88.9059, -87.7018, '1/2', '-', 100, 'inf', 7.75, 0, 8.4, 5.8, 1.1, 0, 7.55, 0.15, 7.7, 1.28, 0),
'Zr90': ('Zr', '90', 'Zirconium', 90, 40, 'NULL', -88.7673, '0', '+', 51.45, 'inf', 6.5, 0, 'NULL', 'NULL', 0, 0, 5.1, 0, 5.1, 0.011, 0.175861),
'Zr91': ('Zr', '91', 'Zirconium', 91, 40, 'NULL', -87.8904, '5/2', '+', 11.22, 'inf', 8.8, 0, 7.9, 10.1, -1.08, 0, 9.5, 0.15, 9.65, 1.17, 0.510565),
'Zr92': ('Zr', '92', 'Zirconium', 92, 40, 'NULL', -88.4539, '0', '+', 17.15, 'inf', 7.52, 0, 'NULL', 'NULL', 0, 0, 6.9, 0, 6.9, 0.22, 0.103087),
'Zr94': ('Zr', '94', 'Zirconium', 94, 40, 'NULL', -87.2668, '0', '+', 17.38, 'inf', 8.3, 0, 'NULL', 'NULL', 0, 0, 8.4, 0, 8.4, 0.0499, 0.343786),
'Zr96': ('Zr', '96', 'Zirconium', 96, 40, 'NULL', -85.4428, '0', '+', 2.8, 'inf', 5.5, 0, 'NULL', 'NULL', 0, 0, 3.8, 0, 3.8, 0.0229, 0.409936),
'Zr': ('Zr', 'nat', 'Zirconium', 91, 40, 91.224, 0, 'NULL', 'NULL', 100, 'inf', 7.16, 0, 'NULL', 'NULL', 'NULL', 'NULL', 6.44, 0.02, 6.46, 0.185, 0),
'Nb': ('Nb', 'nat', 'Niobium', 93, 41, 92.9064, -87.2083, '9/2', '+', 100, 'inf', 7.054, 0, 7.06, 7.35, -0.139, 0, 6.253, 0.0024, 6.2554, 1.15, 0),
'Mo92': ('Mo', '92', 'Molybdenum', 92, 42, 91.9068, -86.805, '0', '+', 14.84, 'inf', 6.93, 0, 'NULL', 'NULL', 0, 0, 6, 0, 6, 0.019, 0.0666488),
'Mo94': ('Mo', '94', 'Molybdenum', 94, 42, 93.9051, -88.4097, '0', '+', 9.25, 'inf', 6.82, 0, 'NULL', 'NULL', 0, 0, 5.81, 0, 5.81, 0.015, 0.0330556),
'Mo95': ('Mo', '95', 'Molybdenum', 95, 42, 94.9058, -87.7075, '5/2', '+', 15.92, 'inf', 6.93, 0, 'NULL', 'NULL', 'NULL', 'NULL', 6, 0.5, 6.5, 13.1, 0.0666488),
'Mo96': ('Mo', '96', 'Molybdenum', 96, 42, 95.9047, -88.7905, '0', '+', 16.68, 'inf', 6.22, 0, 'NULL', 'NULL', 0, 0, 4.83, 0, 4.83, 0.5, 0.140718),
'Mo97': ('Mo', '97', 'Molybdenum', 97, 42, 96.906, -87.5404, '5/2', '+', 9.55, 'inf', 7.26, 0, 'NULL', 'NULL', 'NULL', 'NULL', 6.59, 0.5, 7.1, 2.5, 0.170653),
'Mo98': ('Mo', '98', 'Molybdenum', 98, 42, 97.9054, -88.1117, '0', '+', 24.13, 'inf', 6.6, 0, 'NULL', 'NULL', 0, 0, 5.44, 0, 5.44, 0.127, 0.0325181),
'Mo100': ('Mo', '100', 'Molybdenum', 100, 42, 99.9075, -86.1843, '0', '+', 9.63, 'inf', 6.75, 0, 'NULL', 'NULL', 0, 0, 5.69, 0, 5.69, 0.4, 0.011958),
'Mo': ('Mo', 'nat', 'Molybdenum', 96, 42, 95.96, 'NULL', 'NULL', 'NULL', 100, 'inf', 6.71, 0, 'NULL', 'NULL', 'NULL', 'NULL', 5.67, 0.04, 5.71, 2.48, 0),
'Tc': ('Tc', 'nat', 'Technecium', 99, 43, 98, -87.3231, '9/2', '+', 100, '2.111E+5y', 6.8, 0, 'NULL', 'NULL', 'NULL', 'NULL', 5.8, 0.5, 6.3, 20, 0),
'Ru96': ('Ru', '96', 'Ruthenium', 96, 44, 95.9076, -86.0721, '0', '+', 5.54, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 0, 0, 'NULL', 0, 'NULL', 0.28, 1),
'Ru98': ('Ru', '98', 'Ruthenium', 98, 44, 97.9053, -88.2245, '0', '+', 1.87, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 0, 0, 'NULL', 0, 'NULL', 0.8, 1),
'Ru99': ('Ru', '99', 'Ruthenium', 99, 44, 98.9059, -87.617, '5/2', '+', 12.76, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 6.9, 1),
'Ru100': ('Ru', '100', 'Ruthenium', 100, 44, 99.9042, -89.219, '0', '+', 12.6, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 0, 0, 'NULL', 0, 'NULL', 4.8, 1),
'Ru101': ('Ru', '101', 'Ruthenium', 101, 44, 100.906, -87.9497, '5/2', '+', 17.06, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 3.3, 1),
'Ru102': ('Ru', '102', 'Ruthenium', 102, 44, 101.904, -89.098, '0', '+', 31.55, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 0, 0, 'NULL', 0, 'NULL', 1.17, 1),
'Ru104': ('Ru', '104', 'Ruthenium', 104, 44, 103.905, -88.0889, '0', '+', 18.62, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 0, 0, 'NULL', 0, 'NULL', 0.31, 1),
'Ru': ('Ru', 'nat', 'Ruthenium', 101, 44, 101.07, 'NULL', 'NULL', 'NULL', 100, 'inf', 7.02, 0, 'NULL', 'NULL', 'NULL', 'NULL', 6.21, 0.4, 6.6, 2.56, 0),
'Rh': ('Rh', 'nat', 'Rhodium', 103, 45, 102.906, -88.0222, '1/2', '-', 100, 'inf', 5.9, 0, 8.15, 6.74, 0.614, 0, 4.34, 0.047, 4.39, 144.8, 0),
'Pd102': ('Pd', '102', 'Palladium', 102, 46, 101.906, -87.9251, '0', '+', 1.02, 'inf', 7.7, 0, 'NULL', 'NULL', 0, 0, 7.5, 0, 7.5, 3.4, 0.697487),
'Pd104': ('Pd', '104', 'Palladium', 104, 46, 103.904, -89.39, '0', '+', 11.14, 'inf', 7.7, 0, 'NULL', 'NULL', 0, 0, 7.5, 0, 7.5, 0.6, 0.697487),
'Pd105': ('Pd', '105', 'Palladium', 105, 46, 104.905, -88.4128, '5/2', '+', 22.33, 'inf', 5.5, 0, 'NULL', 'NULL', -2.6, 0, 3.8, 0.8, 4.6, 20, 0.133935),
'Pd106': ('Pd', '106', 'Palladium', 106, 46, 105.903, -89.9025, '0', '+', 27.33, 'inf', 6.4, 0, 'NULL', 'NULL', 0, 0, 5.1, 0, 5.1, 0.304, 0.172695),
'Pd108': ('Pd', '108', 'Palladium', 108, 46, 107.904, -89.5243, '0', '+', 26.46, 'inf', 4.1, 0, 'NULL', 'NULL', 0, 0, 2.1, 0, 2.1, 8.5, 0.518726),
'Pd110': ('Pd', '110', 'Palladium', 110, 46, 109.905, -88.3492, '0', '+', 11.72, 'inf', 7.7, 0, 'NULL', 'NULL', 0, 0, 7.5, 0, 7.5, 0.226, 0.697487),
'Pd': ('Pd', 'nat', 'Palladium', 106, 46, 106.42, 'NULL', 'NULL', 'NULL', 100, 'inf', 5.91, 0, 'NULL', 'NULL', 'NULL', 'NULL', 4.39, 0.09, 4.48, 6.9, 0),
'Ag107': ('Ag', '107', 'Silver', 107, 47, 'NULL', -88.4017, '1/2', '-', 51.839, 'inf', 7.555, 0, 8.14, 5.8, 1, 0, 7.17, 0.13, 7.3, 37.6, 0.627542),
'Ag109': ('Ag', '109', 'Silver', 109, 47, 'NULL', -88.7227, '1/2', '-', 48.161, 'inf', 4.165, 0, 3.24, 6.9, -1.6, 0, 2.18, 0.32, 2.5, 91, 0.505355),
'Ag': ('Ag', 'nat', 'Silver', 108, 47, 107.868, 0, 'NULL', 'NULL', 100, 'inf', 5.922, 0, 'NULL', 'NULL', 'NULL', 'NULL', 4.407, 0.58, 4.987, 63.3, 0),
'Cd106': ('Cd', '106', 'Cadmium', 106, 48, 105.906, -87.1325, '0', '+', 1.25, 'inf', 5, 0, 'NULL', 'NULL', 0, 0, 3.1, 0, 3.1, 1, 0.0495867),
'Cd108': ('Cd', '108', 'Cadmium', 108, 48, 107.904, -89.2523, '0', '+', 0.89, 'inf', 5.31, 0, 'NULL', 'NULL', 0, 0, 3.7, 0, 3.7, 1.1, 0.18377),
'Cd110': ('Cd', '110', 'Cadmium', 110, 48, 109.903, -90.353, '0', '+', 12.49, 'inf', 5.78, 0, 'NULL', 'NULL', 0, 0, 4.4, 0, 4.4, 11, 0.4026),
'Cd111': ('Cd', '111', 'Cadmium', 111, 48, 110.904, -89.2575, '1/2', '+', 12.8, 'inf', 6.47, 0, 'NULL', 'NULL', 'NULL', 'NULL', 5.3, 0.3, 5.6, 24, 0.757466),
'Cd112': ('Cd', '112', 'Cadmium', 112, 48, 111.903, -90.5805, '0', '+', 24.13, 'inf', 6.34, 0, 'NULL', 'NULL', 0, 0, 5.1, 0, 5.1, 2.2, 0.687551),
'Cd113': ('Cd', '113', 'Cadmium', 113, 48, 112.904, -89.0493, '1/2', '+', 12.22, 'inf', -8, 5.73, 'NULL', 'NULL', 'NULL', 'NULL', 12.1, 0.3, 12.4, 20600, 3.06538),
'Cd114': ('Cd', '114', 'Cadmium', 114, 48, 113.903, -90.0209, '0', '+', 28.73, 'inf', 7.48, 0, 'NULL', 'NULL', 0, 0, 7.1, 0, 7.1, 0.34, 1.34899),
'Cd116': ('Cd', '116', 'Cadmium', 116, 48, 115.905, -88.7194, '0', '+', 7.49, 'inf', 6.26, 0, 'NULL', 'NULL', 0, 0, 5, 0, 5, 0.075, 0.645231),
'Cd': ('Cd', 'nat', 'Cadmium', 112, 48, 112.411, 'NULL', 'NULL', 'NULL', 100, 'inf', 4.83, -0.7, 'NULL', 'NULL', 'NULL', 'NULL', 3.04, 3.46, 6.5, 2520, 0),
'In113': ('In', '113', 'Indium', 113, 49, 112.904, -89.3696, '9/2', '+', 4.29, 'inf', 5.39, 0, 'NULL', 'NULL', 'NULL', 'NULL', 3.65, 0.000037, 3.65, 12, 0.757843),
'In115': ('In', '115', 'Indium', 115, 49, 114.904, -89.5366, '9/2', '+', 95.71, 'inf', 4, -0.0562, 2.1, 6.4, -2.1, 0, 2.02, 0.55, 2.57, 202, 0.0317037),
'In': ('In', 'nat', 'Indium', 115, 49, 114.818, 'NULL', 'NULL', 'NULL', 100, 'inf', 4.065, -0.0539, 'NULL', 'NULL', 'NULL', 'NULL', 2.08, 0.54, 2.62, 193.8, 0),
'Sn112': ('Sn', '112', 'Tin', 112, 50, 111.905, -88.6613, '0', '+', 0.97, 'inf', 6, 0, 'NULL', 'NULL', 0, 0, 4.5, 0, 4.5, 1, 0.0709827),
'Sn114': ('Sn', '114', 'Tin', 114, 50, 113.903, -90.5609, '0', '+', 0.66, 'inf', 6, 0, 'NULL', 'NULL', 0, 0, 4.8, 0, 4.8, 0.114, 0.0709827),
'Sn115': ('Sn', '115', 'Tin', 115, 50, 114.903, -90.036, '1/2', '+', 0.34, 'inf', 6, 0, 'NULL', 'NULL', 'NULL', 'NULL', 4.5, 0.3, 4.8, 30, 0.0709827),
'Sn116': ('Sn', '116', 'Tin', 116, 50, 115.902, -91.5281, '0', '+', 14.54, 'inf', 6.1, 0, 'NULL', 'NULL', 0, 0, 4.42, 0, 4.42, 0.14, 0.0397574),
'Sn117': ('Sn', '117', 'Tin', 117, 50, 116.903, -90.4, '1/2', '+', 7.68, 'inf', 6.59, 0, 0.22, -0.23, 0.19, 0, 5.28, 0.3, 5.6, 2.3, 0.120707),
'Sn118': ('Sn', '118', 'Tin', 118, 50, 117.902, -91.6561, '0', '+', 24.22, 'inf', 6.23, 0, 'NULL', 'NULL', 0, 0, 4.63, 0, 4.63, 0.22, 0.00160707),
'Sn119': ('Sn', '119', 'Tin', 119, 50, 118.903, -90.0684, '1/2', '+', 8.59, 'inf', 6.28, 0, 0.14, 0, 0.06, 0, 4.71, 0.3, 5, 2.2, 0.0177487),
'Sn120': ('Sn', '120', 'Tin', 120, 50, 119.902, -91.1051, '0', '+', 32.58, 'inf', 6.67, 0, 'NULL', 'NULL', 0, 0, 5.29, 0, 5.29, 0.14, 0.148082),
'Sn122': ('Sn', '122', 'Tin', 122, 50, 121.903, -89.946, '0', '+', 4.63, 'inf', 5.93, 0, 'NULL', 'NULL', 0, 0, 4.14, 0, 4.14, 0.18, 0.0925333),
'Sn124': ('Sn', '124', 'Tin', 124, 50, 123.905, -88.2367, '0', '+', 5.79, 'inf', 5.79, 0, 'NULL', 'NULL', 0, 0, 4.48, 0, 4.48, 0.133, 0.134876),
'Sn': ('Sn', 'nat', 'Tin', 119, 50, 118.71, 'NULL', 'NULL', 'NULL', 100, 'inf', 6.225, 0, 'NULL', 'NULL', 'NULL', 'NULL', 4.871, 0.022, 4.892, 0.626, 0),
'Sb121': ('Sb', '121', 'Antimony', 121, 51, 'NULL', -89.5951, '5/2', '+', 57.21, 'inf', 5.71, 0, 5.7, 5.8, -0.05, 'NULL', 4.1, 0.0003, 4.1003, 5.75, 0.0509011),
'Sb123': ('Sb', '123', 'Antimony', 123, 51, 'NULL', -89.2241, '7/2', '+', 42.79, 'inf', 5.38, 0, 5.2, 5.4, -0.1, 0, 3.64, 0.001, 3.641, 3.8, 0.067059),
'Sb': ('Sb', 'nat', 'Antimony', 122, 51, 121.76, 0, 'NULL', 'NULL', 100, 'inf', 5.57, 0, 'NULL', 'NULL', 'NULL', 'NULL', 3.9, 0, 3.9, 4.91, 0),
'Te120': ('Te', '120', 'Tellurium', 120, 52, 119.904, -89.4046, '0', '+', 0.09, 'inf', 5.3, 0, 'NULL', 'NULL', 0, 0, 3.5, 0, 3.5, 2.3, 0.129327),
'Te122': ('Te', '122', 'Tellurium', 122, 52, 121.903, -90.314, '0', '+', 2.55, 'inf', 3.8, 0, 'NULL', 'NULL', 0, 0, 1.8, 0, 1.8, 3.4, 0.55242),
'Te123': ('Te', '123', 'Tellurium', 123, 52, 122.904, -89.1719, '1/2', '+', 0.89, 'inf', -0.05, -0.116, -1.2, 3.5, -2.04, 0, 0.002, 0.52, 0.52, 418, 0.999505),
'Te124': ('Te', '124', 'Tellurium', 124, 52, 123.903, -90.5245, '0', '+', 4.74, 'inf', 7.95, 0, 'NULL', 'NULL', 0, 0, 8, 0, 8, 6.8, 0.959014),
'Te125': ('Te', '125', 'Tellurium', 125, 52, 124.904, -89.0222, '1/2', '+', 7.07, 'inf', 5.01, 0, 4.9, 5.2, -0.26, 0, 3.17, 0.008, 3.18, 1.55, 0.222001),
'Te126': ('Te', '126', 'Tellurium', 126, 52, 125.903, -90.0646, '0', '+', 18.84, 'inf', 5.55, 0, 'NULL', 'NULL', 0, 0, 3.88, 0, 3.88, 1.04, 0.0452508),
'Te128': ('Te', '128', 'Tellurium', 128, 52, 127.904, -88.9921, '0', '+', 31.74, 'inf', 5.88, 0, 'NULL', 'NULL', 0, 0, 4.36, 0, 4.36, 0.215, 0.0716624),
'Te130': ('Te', '130', 'Tellurium', 130, 52, 129.906, -87.3514, '0', '+', 34.08, 'inf', 6.01, 0, 'NULL', 'NULL', 0, 0, 4.55, 0, 4.55, 0.29, 0.119573),
'Te': ('Te', 'nat', 'Tellurium', 128, 52, 127.6, 0, 'NULL', 'NULL', 100, 'inf', 5.68, 0, 'NULL', 'NULL', 'NULL', 'NULL', 4.23, 0.09, 4.32, 4.7, 0),
'I': ('I', 'nat', 'Iodine', 127, 53, 126.904, -88.9831, '5/2', '+', 100, 'inf', 5.28, 0, 6.6, 3.4, 1.58, 0, 3.5, 0.31, 3.81, 6.15, 0),
'Xe124': ('Xe', '124', 'Xenon', 124, 54, 123.906, -87.6601, '0', '+', 0.0952, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 0, 0, 'NULL', 0, 'NULL', 165, 1),
'Xe126': ('Xe', '126', 'Xenon', 126, 54, 125.904, -89.1685, '0', '+', 0.089, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 0, 0, 'NULL', 0, 'NULL', 3.5, 1),
'Xe128': ('Xe', '128', 'Xenon', 128, 54, 127.904, -89.86, '0', '+', 1.9102, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 0, 0, 'NULL', 0, 'NULL', 8, 1),
'Xe129': ('Xe', '129', 'Xenon', 129, 54, 128.905, -88.6974, '1/2', '+', 26.4006, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 21, 1),
'Xe130': ('Xe', '130', 'Xenon', 130, 54, 129.904, -89.8817, '0', '+', 4.071, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 0, 0, 'NULL', 0, 'NULL', 26, 1),
'Xe131': ('Xe', '131', 'Xenon', 131, 54, 130.905, -88.4152, '3/2', '+', 21.332, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 85, 1),
'Xe132': ('Xe', '132', 'Xenon', 132, 54, 131.904, -89.2805, '0', '+', 26.9086, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 0, 0, 'NULL', 0, 'NULL', 0.45, 1),
'Xe134': ('Xe', '134', 'Xenon', 134, 54, 133.905, -88.1245, '0', '+', 10.4357, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 0, 0, 'NULL', 0, 'NULL', 0.265, 1),
'Xe136': ('Xe', '136', 'Xenon', 136, 54, 135.907, -86.4251, '0', '+', 8.8573, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 0, 0, 'NULL', 0, 'NULL', 0.26, 1),
'Xe': ('Xe', 'nat', 'Xenon', 131, 54, 131.293, 'NULL', 'NULL', 'NULL', 100, 'inf', 4.69, 0, 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 0, 'NULL', 165, 0),
'Cs': ('Cs', 'nat', 'Caesium', 133, 55, 132.905, -88.071, '7/2', '+', 100, 'inf', 5.42, 0, 'NULL', 'NULL', 1.29, 0, 3.69, 0.21, 3.9, 29, 0),
'Ba130': ('Ba', '130', 'Barium', 130, 56, 129.906, -87.2616, '0', '+', 0.106, 'NULL', -3.6, 0, 'NULL', 'NULL', 0, 0, 1.6, 0, 1.6, 30, 'NULL'),
'Ba132': ('Ba', '132', 'Barium', 132, 56, 131.905, -88.4348, '0', '+', 0.101, 'inf', 7.8, 0, 'NULL', 'NULL', 0, 0, 7.6, 0, 7.6, 7, 'NULL'),
'Ba134': ('Ba', '134', 'Barium', 134, 56, 133.904, -88.9499, '0', '+', 2.417, 'inf', 5.7, 0, 'NULL', 'NULL', 0, 0, 4.08, 0, 4.08, 2, 'NULL'),
'Ba135': ('Ba', '135', 'Barium', 135, 56, 134.906, -87.8505, '3/2', '+', 6.592, 'inf', 4.66, 0, 'NULL', 'NULL', 'NULL', 'NULL', 2.74, 0.5, 3.2, 5.8, 'NULL'),
'Ba136': ('Ba', '136', 'Barium', 136, 56, 135.905, -88.8869, '0', '+', 7.854, 'inf', 4.9, 0, 'NULL', 'NULL', 0, 0, 3.03, 0, 3.03, 0.68, 'NULL'),
'Ba137': ('Ba', '137', 'Barium', 137, 56, 136.906, -87.7212, '3/2', '+', 11.232, 'inf', 6.82, 0, 'NULL', 'NULL', 'NULL', 'NULL', 5.86, 0.5, 6.4, 3.6, 'NULL'),
'Ba138': ('Ba', '138', 'Barium', 138, 56, 137.905, -88.2616, '0', '+', 1.6987, 'inf', 4.83, 0, 'NULL', 'NULL', 0, 0, 2.94, 0, 2.94, 0.27, 'NULL'),
'Ba': ('Ba', 'nat', 'Barium', 137, 56, 137.327, 'NULL', 'NULL', 'NULL', 100, 'inf', 5.07, 0, 'NULL', 'NULL', 0, 0, 3.23, 0.15, 3.38, 1.1, 0),
'La138': ('La', '138', 'Lanthanum', 138, 57, 137.907, -86.5247, '5', '+', 0.08881, 'inf', 8, 0, 'NULL', 'NULL', 'NULL', 'NULL', 8, 0.5, 8.5, 57, 0.0574041),
'La139': ('La', '139', 'Lanthanum', 139, 57, 138.906, -87.2314, '7/2', '+', 99.9119, 'inf', 8.24, 0, 11.4, 4.5, 3, 0, 8.53, 11.13, 9.66, 8.93, 0),
'La': ('La', 'nat', 'Lanthanum', 139, 57, 138.905, 'NULL', 'NULL', 'NULL', 100, 'inf', 8.24, 0, 'NULL', 'NULL', 'NULL', 'NULL', 8.53, 1.13, 9.66, 8.97, 0),
'Ce136': ('Ce', '136', 'Cerium', 136, 58, 135.907, -86.4683, '0', '+', 0.185, 'inf', 5.76, 0, 'NULL', 'NULL', 0, 0, 4.23, 0, 4.23, 7.3, 0.416297),
'Ce138': ('Ce', '138', 'Cerium', 138, 58, 137.906, -87.5685, '0', '+', 0.251, 'inf', 6.65, 0, 'NULL', 'NULL', 0, 0, 5.64, 0, 5.64, 1.1, 0.887785),
'Ce140': ('Ce', '140', 'Cerium', 140, 58, 139.905, -88.0833, '0', '+', 88.45, 'inf', 4.81, 0, 'NULL', 'NULL', 0, 0, 2.94, 0, 2.94, 0.57, 0.0123583),
'Ce142': ('Ce', '142', 'Cerium', 142, 58, 141.909, -84.5385, '0', '+', 11.114, 'inf', 4.72, 0, 'NULL', 'NULL', 0, 0, 2.84, 0, 2.84, 0.95, 0.0489721),
'Ce': ('Ce', 'nat', 'Cerium', 140, 58, 140.116, 'NULL', 'NULL', 'NULL', 100, 'inf', 4.84, 0, 'NULL', 'NULL', 'NULL', 'NULL', 2.94, 0, 2.94, 0.63, 0),
'Pr': ('Pr', 'nat', 'Praseodymium', 141, 59, 140.908, -86.0209, '5/2', '+', 100, 'inf', 4.58, 0, 'NULL', 'NULL', -0.055, 0, 2.64, 0.01, 2.66, 11.5, 0),
'Nd142': ('Nd', '142', 'Neodymium', 142, 60, 141.908, -85.9552, '0', '+', 27.152, 'inf', 7.7, 0, 'NULL', 'NULL', 0, 0, 7.5, 0, 7.5, 18.7, 0.00260247),
'Nd143': ('Nd', '143', 'Neodymium', 143, 60, 142.91, -84.0074, '7/2', '+', 12.174, 'inf', 14, 0, 'NULL', 'NULL', -21, 0, 25, 55, 80, 337, 2.31439),
'Nd144': ('Nd', '144', 'Neodymium', 144, 60, 143.91, -83.7532, '0', '+', 23.798, 'inf', 2.8, 0, 'NULL', 'NULL', 0, 0, 1, 0, 1, 3.6, 0.867424),
'Nd145': ('Nd', '145', 'Neodymium', 145, 60, 144.913, -81.4371, '7/2', '+', 8.293, 'inf', 14, 0, 'NULL', 'NULL', 'NULL', 'NULL', 25, 5, 30, 42, 2.31439),
'Nd146': ('Nd', '146', 'Neodymium', 146, 60, 145.913, -80.931, '0', '+', 17.189, 'inf', 8.7, 0, 'NULL', 'NULL', 0, 0, 9.5, 0, 9.5, 1.4, 0.279929),
'Nd148': ('Nd', '148', 'Neodymium', 148, 60, 147.917, -77.4134, '0', '+', 5.756, 'inf', 5.7, 0, 'NULL', 'NULL', 0, 0, 4.1, 0, 4.1, 2.5, 0.450589),
'Nd150': ('Nd', '150', 'Neodymium', 150, 60, 149.921, -73.6897, '0', '+', 5.638, 'inf', 5.28, 5.28, 'NULL', 'NULL', 0, 0, 3.5, 0, 3.5, 1.2, 0.0571445),
'Nd': ('Nd', 'nat', 'Neodymium', 144, 60, 144.242, 'NULL', 'NULL', 'NULL', 100, 'inf', 7.69, 0, 'NULL', 'NULL', 'NULL', 'NULL', 7.43, 9.2, 16.6, 50.5, 0),
'Pm': ('Pm', 'nat', 'Promethium', 147, 61, 'NULL', -79.0479, '7/2', '+', 100, '2.6234y', 12.6, 0, 'NULL', 'NULL', 'NULL', 'NULL', 20, 1.3, 21.3, 168.4, 0),
'Sm144': ('Sm', '144', 'Samarium', 144, 62, 143.912, -81.972, '0', '+', 3.07, 'inf', -3, 0, 'NULL', 'NULL', 0, 0, 1, 0, 1, 0.7, 2.30579),
'Sm147': ('Sm', '147', 'Samarium', 147, 62, 146.915, -79.2721, '7/2', '-', 14.99, 'inf', 14, 0, 'NULL', 'NULL', -11, 0, 25, 14, 39, 57, 70.9927),
'Sm148': ('Sm', '148', 'Samarium', 148, 62, 147.915, -79.3422, '0', '+', 11.24, 'inf', -3, 0, 'NULL', 'NULL', 0, 0, 1, 0, 1, 2.4, 2.30579),
'Sm149': ('Sm', '149', 'Samarium', 149, 62, 148.917, -77.1419, '7/2', '-', 13.82, 'inf', 18.7, -11.7, 'NULL', 'NULL', -31.4, -10.3, 63.5, 137, 200, 42080, 177.725),
'Sm150': ('Sm', '150', 'Samarium', 150, 62, 149.917, -77.0573, '0', '+', 7.38, 'inf', 14, 0, 'NULL', 'NULL', 0, 0, 25, 0, 25, 104, 70.9927),
'Sm152': ('Sm', '152', 'Samarium', 152, 62, 151.92, -74.7688, '0', '+', 26.75, 'inf', -5, 0, 'NULL', 'NULL', 0, 0, 3.1, 0, 3.1, 206, 8.18274),
'Sm154': ('Sm', '154', 'Samarium', 154, 62, 153.922, -72.4616, '0', '+', 22.75, 'inf', -8, 0, 'NULL', 'NULL', 0, 0, 11, 0, 11, 8.4, 22.5078),
'Sm': ('Sm', 'nat', 'Samarium', 150, 62, 150.36, 'NULL', 'NULL', 'NULL', 100, 'inf', 0, -1.65, 'NULL', 'NULL', 'NULL', 'NULL', 0.422, 39, 39.4, 5922, 0),
'Eu151': ('Eu', '151', 'Europium', 151, 63, 150.92, -74.6591, '5/2', '+', 47.81, 'inf', 6.92, 2.53, 'NULL', 'NULL', -4.5, 2.14, 5.5, 3.1, 8.6, 9100, 0.829235),
'Eu153': ('Eu', '153', 'Europium', 153, 63, 152.921, -73.3735, '5/2', '+', 52.19, 'inf', 8.22, 0, 'NULL', 'NULL', -3.2, 0, 8.5, 1.3, 9.8, 312, 1.27675),
'Eu': ('Eu', 'nat', 'Europium', 152, 63, 151.964, 'NULL', 'NULL', 'NULL', 100, 'inf', 5.3, -1.26, 'NULL', 'NULL', 'NULL', 'NULL', 6.57, 2.5, 9.2, 4530, 0),
'Gd152': ('Gd', '152', 'Gadolinium', 152, 64, 151.92, -74.7142, '0', '+', 0.2, 'inf', 10, 0, 'NULL', 'NULL', 0, 0, 13, 0, 13, 735, 0.644435),
'Gd154': ('Gd', '154', 'Gadolinium', 154, 64, 153.921, -73.7132, '0', '+', 2.18, 'inf', 10, 0, 'NULL', 'NULL', 0, 0, 13, 0, 13, 85, 0.644435),
'Gd155': ('Gd', '155', 'Gadolinium', 155, 64, 154.923, -72.0771, '3/2', '-', 14.8, 'inf', 13.8, -17, 'NULL', 'NULL', -5, -13.16, 40.8, 25, 66, 61100, 0.704722),
'Gd156': ('Gd', '156', 'Gadolinium', 156, 64, 155.922, -72.5422, '0', '+', 20.47, 'inf', 6.3, 0, 'NULL', 'NULL', 0, 0, 5, 0, 5, 1.5, 0.858876),
'Gd157': ('Gd', '157', 'Gadolinium', 157, 64, 156.924, -70.8307, '3/2', '-', 15.65, 'inf', -1.14, -72, 'NULL', 'NULL', 5, -55.8, 650, 349, 104.4, 259000, 17.4371),
'Gd158': ('Gd', '158', 'Gadolinium', 158, 64, 157.924, -70.6967, '0', '+', 24.84, 'inf', 9, 0, 'NULL', 'NULL', 0, 0, 10, 0, 10, 2.2, 0.711992),
'Gd160': ('Gd', '160', 'Gadolinium', 160, 64, 159.927, -67.9486, '0', '+', 21.58, 'inf', 9.15, 0, 'NULL', 'NULL', 0, 0, 10.52, 0, 10.52, 0.77, 0.702312),
'Gd': ('Gd', 'nat', 'Gadolinium', 157, 64, 157.25, 'NULL', 'NULL', 'NULL', 100, 'inf', 9.5, -13.82, 'NULL', 'NULL', 'NULL', 'NULL', 29.3, 151, 180, 59700, 0),
'Tb': ('Tb', 'nat', 'Terbium', 159, 65, 158.925, -69.539, '3/2', '+', 100, 'inf', 7.34, 0, 6.8, 8.1, -0.17, 0, 6.48, 0.004, 6.48, 23.4, 0),
'Dy156': ('Dy', '156', 'Dysprosium', 156, 66, 155.924, -70.5298, '0', '+', 0.056, 'inf', 6.1, 0, 'NULL', 'NULL', 0, 0, 4.7, 0, 4.7, 33, 0.869752),
'Dy158': ('Dy', '158', 'Dysprosium', 158, 66, 157.924, -70.4121, '0', '+', 0.095, 'inf', 6.7, 0, 'NULL', 'NULL', 0, 0, 5, 0, 5, 43, 0.84287),
'Dy160': ('Dy', '160', 'Dysprosium', 160, 66, 159.925, -69.6781, '0', '+', 2.329, 'inf', 6.7, 0, 'NULL', 'NULL', 0, 0, 5.6, 0, 5.6, 56, 0.84287),
'Dy161': ('Dy', '161', 'Dysprosium', 161, 66, 160.927, -68.0611, '5/2', '+', 18.889, 'inf', 10.3, 0, 14.5, 4.2, -0.17, 0, 13.3, 3, 16, 600, 0.628648),
'Dy162': ('Dy', '162', 'Dysprosium', 162, 66, 161.927, -68.1868, '0', '+', 25.475, 'inf', -1.4, 0, 'NULL', 'NULL', 0, 0, 0.25, 0, 0.25, 194, 0.993139),
'Dy163': ('Dy', '163', 'Dysprosium', 163, 66, 162.929, -66.3865, '5/2', '-', 24.896, 'inf', 5, 0, 6.1, 3.5, 1.3, 0, 3.1, 0.21, 3.3, 124, 0.912491),
'Dy164': ('Dy', '164', 'Dysprosium', 164, 66, 163.929, -65.9733, '0', '+', 28.26, 'inf', 49.4, -0.79, 'NULL', 'NULL', 0, 0, 307, 0, 307, 2840, 7.54428),
'Dy': ('Dy', 'nat', 'Dysprosium', 163, 66, 162.5, 'NULL', 'NULL', 'NULL', 100, 'inf', 16.9, 0.276, 'NULL', 'NULL', 'NULL', 'NULL', 35.9, 54.4, 90.3, 994, 0),
'Ho': ('Ho', 'nat', 'Holmium', 165, 67, 164.93, -64.9046, '7/2', '-', 100, 'inf', 8.44, 0, 6.9, 10.3, -1.69, 0, 8.06, 0.36, 8.42, 64.7, 0),
'Er162': ('Er', '162', 'Erbium', 162, 68, 161.929, -66.3426, '0', '+', 0.139, 'inf', 9.01, 0, 'NULL', 'NULL', 0, 0, 9.7, 0, 9.7, 19, 0.337749),
'Er164': ('Er', '164', 'Erbium', 164, 68, 163.929, -65.9496, '0', '+', 1.601, 'inf', 7.95, 0, 'NULL', 'NULL', 0, 0, 8.4, 0, 8.4, 13, 0.0415002),
'Er166': ('Er', '166', 'Erbium', 166, 68, 165.93, -64.9316, '0', '+', 33.503, 'inf', 10.51, 0, 'NULL', 'NULL', 0, 0, 14.1, 0, 14.1, 19.6, 0.820248),
'Er167': ('Er', '167', 'Erbium', 167, 68, 166.932, -63.2967, '7/2', '+', 22.869, 'inf', 3.06, 0, 5.3, 0, 2.6, 0, 1.1, 0.13, 1.2, 659, 0.845699),
'Er169': ('Er', '168', 'Erbium', 168, 68, 167.932, -62.9967, '0', '+', 26.978, 'inf', 7.43, 0, 'NULL', 'NULL', 0, 0, 6.9, 0, 6.9, 2.74, 0.0902905),
'Er170': ('Er', '170', 'Erbium', 170, 68, 169.935, -60.1146, '0', '+', 14.91, 'inf', 9.61, 0, 'NULL', 'NULL', 0, 0, 11.6, 0, 11.6, 5.8, 0.52185),
'Er': ('Er', 'nat', 'Erbium', 167, 68, 167.259, 'NULL', 'NULL', 'NULL', 100, 'inf', 7.79, 0, 'NULL', 'NULL', 'NULL', 'NULL', 7.63, 1.1, 8.7, 159, 0),
'Tm': ('Tm', 'nat', 'Thulium', 169, 69, 168.934, -61.28, '1/2', '+', 100, 'inf', 9.61, 0, 'NULL', 'NULL', 0.9, 0, 6.28, 0.1, 6.38, 100, 0),
'Yb168': ('Yb', '168', 'Ytterbium', 168, 70, 167.934, -61.5746, '0', '+', 0.123, 'inf', -4.07, -0.62, 'NULL', 'NULL', 0, 0, 2.13, 0, 2.13, 2230, 0.889945),
'Yb170': ('Yb', '170', 'Ytterbium', 170, 70, 169.935, -60.769, '0', '+', 2.982, 'inf', 6.8, 0, 'NULL', 'NULL', 0, 0, 5.8, 0, 5.8, 11.4, 0.699756),
'Yb171': ('Yb', '171', 'Ytterbium', 171, 70, 170.936, -59.3121, '1/2', '-', 14.09, 'inf', 9.7, 0, 9.5, 19.4, -5.59, 0, 11.7, 3.9, 15.6, 48.6, 0.389058),
'Yb172': ('Yb', '172', 'Ytterbium', 172, 70, 171.936, -59.2603, '0', '+', 21.68, 'inf', 9.5, 0, 'NULL', 'NULL', 0, 0, 11.2, 0, 11.2, 0.8, 0.413992),
'Yb173': ('Yb', '173', 'Ytterbium', 173, 70, 172.938, -57.5563, '5/2', '-', 16.103, 'inf', 9.56, 0, 2.5, 13.3, -5.3, 0, 11.5, 3.5, 15, 17.1, 0.406566),
'Yb174': ('Yb', '174', 'Ytterbium', 174, 70, 173.939, -56.9496, '0', '+', 32.026, 'inf', 19.2, 0, 'NULL', 'NULL', 0, 0, 46.8, 0, 46.8, 69.4, 1.39364),
'Yb176': ('Yb', '176', 'Ytterbium', 176, 70, 175.943, -53.4941, '0', '+', 12.996, 'inf', 8.7, 0, 'NULL', 'NULL', 0, 0, 9.6, 0, 9.6, 2.85, 0.508532),
'Yb': ('Yb', 'nat', 'Ytterbium', 173, 70, 173.054, 'NULL', 'NULL', 'NULL', 100, 'inf', 12.41, 0, 'NULL', 'NULL', 'NULL', 'NULL', 19.42, 4, 23.4, 34.8, 0),
'Lu175': ('Lu', '175', 'Lutetium', 175, 71, 174.941, -55.1707, '7/2', '+', 97.401, 'inf', 7.28, 0, 'NULL', 'NULL', -2.2, 0, 6.59, 0.6, 7.2, 21, 0.0195117),
'Lu176': ('Lu', '176', 'Lutetium', 176, 71, 175.943, -53.3874, '7', '-', 2.599, 'inf', 6.1, -0.57, 'NULL', 'NULL', -3, 0.61, 4.7, 1.2, 5.9, 2065, 0.277954),
'Lu': ('Lu', 'nat', 'Lutetium', 175, 71, 174.967, 'NULL', 'NULL', 'NULL', 100, 'inf', 7.21, 0, 'NULL', 'NULL', 'NULL', 'NULL', 6.53, 0.7, 7.2, 74, 0),
'Hf174': ('Hf', '174', 'Hafnium', 174, 72, 173.94, -55.8466, '0', '+', 0.16, 'inf', 10.9, 0, 'NULL', 'NULL', 0, 0, 15, 0, 15, 561, 0.967936),
'Hf176': ('Hf', '176', 'Hafnium', 176, 72, 175.941, -54.5775, '0', '+', 5.26, 'inf', 6.61, 0, 'NULL', 'NULL', 0, 0, 5.5, 0, 5.5, 23.5, 0.276296),
'Hf177': ('Hf', '177', 'Hafnium', 177, 72, 176.943, -52.8896, '7/2', '+', 18.6, 'inf', 0.8, 0, 'NULL', 'NULL', -0.9, 0, 0.1, 0.1, 0.2, 373, 0.989399),
'Hf178': ('Hf', '178', 'Hafnium', 178, 72, 177.944, -52.4443, '0', '+', 27.28, 'inf', 5.9, 0, 'NULL', 'NULL', 0, 0, 4.4, 0, 4.4, 84, 0.423417),
'Hf179': ('Hf', '179', 'Hafnium', 179, 72, 178.946, -50.4719, '9/2', '+', 13.62, 'inf', 7.46, 0, 'NULL', 'NULL', -1.06, 0, 7, 0.14, 7.1, 41, 0.0782023),
'Hf180': ('Hf', '180', 'Hafnium', 180, 72, 179.947, -49.7884, '0', '+', 35.08, 'inf', 13.2, 0, 'NULL', 'NULL', 0, 0, 21.9, 0, 21.9, 13.04, 1.88606),
'Hf': ('Hf', 'nat', 'Hafnium', 180, 72, 178.49, 'NULL', 'NULL', 'NULL', 100, 'inf', 7.77, 0, 'NULL', 'NULL', 'NULL', 'NULL', 7.6, 2.6, 10.2, 104.1, 0),
'Ta180': ('Ta', '180', 'Tantalum', 180, 73, 179.947, -48.8591, '9', '-', 0.01201, 'inf', 7, 0, 'NULL', 'NULL', 'NULL', 'NULL', 6.2, 0.5, 7, 563, 0.0262188),
'Ta181': ('Ta', '181', 'Tantalum', 181, 73, 180.948, -48.4416, '7/2', '+', 99.988, 'inf', 6.91, 0, 'NULL', 'NULL', -0.29, 0, 6, 0.01, 6.01, 20.5, 0),
'Ta': ('Ta', 'nat', 'Tantalum', 181, 73, 180.948, 'NULL', 'NULL', 'NULL', 100, 'inf', 6.91, 0, 'NULL', 'NULL', 'NULL', 'NULL', 6, 0.01, 6.01, 20.6, 0),
'W180': ('W', '180', 'Tungsten', 180, 74, 179.947, -49.6445, '0', '+', 0.12, 'inf', 5, 0, 'NULL', 'NULL', 0, 0, 3, 0, 3, 30, 0.105704),
'W182': ('W', '182', 'Tungsten', 182, 74, 181.948, -48.2475, '0', '+', 26.5, 'inf', 7.04, 0, 'NULL', 'NULL', 0, 0, 6.1, 0, 6.1, 20.75, 1.19202),
'W183': ('W', '183', 'Tungsten', 183, 74, 182.95, -46.367, '1/2', '-', 14.31, 'inf', 6.59, 0, 6.3, 7, -0.3, 0, 5.36, 0.3, 5.7, 10.1, 0.920745),
'W184': ('W', '184', 'Tungsten', 184, 74, 183.951, -45.7073, '0', '+', 30.64, 'inf', 7.55, 0, 'NULL', 'NULL', 0, 0, 7.03, 0, 7.03, 1.7, 1.52112),
'W186': ('W', '186', 'Tungsten', 186, 74, 185.954, -42.5095, '0', '+', 28.43, 'inf', -0.73, 0, 'NULL', 'NULL', 0, 0, 0.065, 0, 0.065, 87.9, 0.976431),
'W': ('W', 'nat', 'Tungsten', 184, 74, 183.84, 'NULL', 'NULL', 'NULL', 100, 'inf', 4.755, 0, 'NULL', 'NULL', 'NULL', 'NULL', 2.97, 1.63, 4.6, 18.3, 0),
'Re185': ('Re', '185', 'Rhenium', 185, 75, 184.953, -43.8222, '5/2', '+', 37.4, 'inf', 9, 0, 'NULL', 'NULL', -2, 0, 10.2, 0.5, 10.7, 112, 0.0430057),
'Re187': ('Re', '187', 'Rhenium', 187, 75, 186.956, -41.2157, '5/2', '+', 62.6, 'inf', 9.3, 0, 'NULL', 'NULL', -2.8, 0, 10.9, 1, 11.9, 76.4, 0.0218573),
'Re': ('Re', 'nat', 'Rhenium', 186, 75, 186.207, 'NULL', 'NULL', 'NULL', 100, 'inf', 9.2, 0, 'NULL', 'NULL', 'NULL', 'NULL', 10.6, 0.9, 11.9, 89.7, 0),
'Os184': ('Os', '184', 'Osmium', 184, 76, 183.952, -44.2561, '0', '+', 0.02, 'inf', 10.2, 0, 'NULL', 'NULL', 0, 0, 13, 0, 13, 3000, 0.0912743),
'Os186': ('Os', '186', 'Osmium', 186, 76, 185.954, -42.9995, '0', '+', 1.59, 'inf', 12, 0, 'NULL', 'NULL', 0, 0, 17, 0, 17, 80, 0.257752),
'Os187': ('Os', '187', 'Osmium', 187, 76, 186.956, -41.2182, '1/2', '-', 1.96, 'inf', 10, 0, 'NULL', 'NULL', 'NULL', 'NULL', 13, 0.3, 13, 320, 0.126561),
'Os188': ('Os', '188', 'Osmium', 188, 76, 187.956, -41.1364, '0', '+', 13.24, 'inf', 7.8, 0, 'NULL', 'NULL', 0, 0, 7.3, 0, 7.3, 4.7, 0.4686),
'Os189': ('Os', '189', 'Osmium', 189, 76, 188.958, -38.9854, '3/2', '-', 16.15, 'inf', 11, 0, 'NULL', 'NULL', 'NULL', 'NULL', 14.4, 0.5, 14.9, 25, 0.0568609),
'Os190': ('Os', '190', 'Osmium', 190, 76, 189.958, -38.7063, '0', '+', 26.26, 'inf', 11.4, 0, 'NULL', 'NULL', 0, 0, 15.2, 0, 15.2, 25, 0.135121),
'Os192': ('Os', '192', 'Osmium', 192, 76, 191.961, -35.8805, '0', '+', 40.78, 'inf', 11.9, 0, 'NULL', 'NULL', 0, 0, 16.6, 0, 16.6, 2, 0.236877),
'Os': ('Os', 'nat', 'Osmium', 190, 76, 190.23, 'NULL', 'NULL', 'NULL', 100, 'inf', 10.7, 0, 'NULL', 'NULL', 'NULL', 'NULL', 14.4, 0.3, 14.7, 16, 0),
'Ir191': ('Ir', '191', 'Iridium', 191, 77, 190.961, -36.7064, '3/2', '+', 37.3, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 954, 1),
'Ir193': ('Ir', '193', 'Iridium', 193, 77, 192.963, -34.5338, '3/2', '+', 62.7, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 111, 1),
'Ir': ('Ir', 'nat', 'Iridium', 192, 77, 192.217, 'NULL', 'NULL', 'NULL', 100, 'inf', 10.6, 0, 'NULL', 'NULL', 'NULL', 'NULL', 14.1, 0, 14, 425, 0),
'Pt190': ('Pt', '190', 'Platinium', 190, 78, 189.96, -37.3234, '0', '+', 0.012, 'inf', 9, 0, 'NULL', 'NULL', 0, 0, 10, 0, 10, 152, 0.121094),
'Pt192': ('Pt', '192', 'Platinium', 192, 78, 191.961, -36.2929, '0', '+', 0.782, 'inf', 9.9, 0, 'NULL', 'NULL', 0, 0, 12.3, 0, 12.3, 10, 0.0634766),
'Pt194': ('Pt', '194', 'Platinium', 194, 78, 193.963, -34.7631, '0', '+', 32.86, 'inf', 10.55, 0, 'NULL', 'NULL', 0, 0, 14, 0, 14, 1.44, 0.207709),
'Pt195': ('Pt', '195', 'Platinium', 195, 78, 194.965, -32.7968, '1/2', '-', 33.78, 'inf', 8.91, 0, 'NULL', 'NULL', 1, 0, 9.8, 0.13, 9.9, 27.5, 0.138584),
'Pt196': ('Pt', '196', 'Platinium', 196, 78, 195.965, -32.6474, '0', '+', 25.21, 'inf', 9.89, 0, 'NULL', 'NULL', 0, 0, 12.3, 0, 12.3, 0.72, 0.0613292),
'Pt198': ('Pt', '198', 'Platinium', 198, 78, 197.968, -29.9077, '0', '+', 7.36, 'inf', 7.8, 0, 'NULL', 'NULL', 0, 0, 7.6, 0, 7.6, 3.66, 0.339844),
'Pt1': ('Pt', 'nat', 'Platinium', 195, 78, 195.084, 'NULL', 'NULL', 'NULL', 100, 'inf', 9.6, 0, 'NULL', 'NULL', 'NULL', 'NULL', 11.58, 0.13, 11.71, 10.3, 0),
'Au': ('Au', 'nat', 'Gold', 197, 79, 196.967, -31.1411, '3/2', '+', 100, 'inf', 7.9, 0, 6.26, 9.9, -1.76, 0, 7.32, 0.43, 7.75, 98.65, 0),
'Hg196': ('Hg', '196', 'Mercury', 196, 80, 195.966, -31.8267, '0', '+', 0.15, 'inf', 30.3, -0.85, 'NULL', 'NULL', 0, 0, 115, 0, 115, 3080, 4.79203),
'Hg198': ('Hg', '198', 'Mercury', 198, 80, 197.967, -30.9544, '0', '+', 9.97, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 0, 0, 'NULL', 0, 'NULL', 2.03, 1),
'Hg199': ('Hg', '199', 'Mercury', 199, 80, 198.968, -29.5471, '1/2', '-', 16.87, 'inf', 16.9, -0.6, 'NULL', 'NULL', -15.5, 0, 36, 30, 66, 2150, 0.802703),
'Hg200': ('Hg', '200', 'Mercury', 200, 80, 199.968, -29.5041, '0', '+', 23.1, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 0, 0, 'NULL', 0, 'NULL', 60, 1),
'Hg201': ('Hg', '201', 'Mercury', 201, 80, 200.97, -27.6633, '3/2', '-', 13.18, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 7.8, 1),
'Hg202': ('Hg', '202', 'Mercury', 202, 80, 201.971, -27.3459, '0', '+', 29.86, 'inf', 11.002, 0, 'NULL', 'NULL', 0, 0, 15.2108, 0, 15.2108, 4.89, 0.236961),
'Hg204': ('Hg', '204', 'Mercury', 204, 80, 203.973, -24.6902, '0', '+', 6.87, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 0, 0, 'NULL', 0, 'NULL', 0.43, 1),
'Hg': ('Hg', 'nat', 'Mercury', 200, 80, 200.59, 'NULL', 'NULL', 'NULL', 100, 'inf', 12.595, 0, 'NULL', 'NULL', 'NULL', 'NULL', 20.24, 6.6, 26.84, 372.3, 0),
'Tl203': ('Tl', '203', 'Thallium', 203, 81, 202.972, -25.7612, '1/2', '+', 29.524, 'inf', 8.51, 0, 9.08, 6.62, 1.06, 0, 6.14, 0.14, 6.28, 11.4, 0.0597012),
'Tl205': ('Tl', '205', 'Thallium', 205, 81, 204.974, -23.8206, '1/2', '+', 70.48, 'inf', 8.87, 0, 5.15, 9.43, -0.242, 0, 11.39, 0.007, 11.4, 0.104, 0.0215368),
'Tl': ('Tl', 'nat', 'Thallium', 204, 81, 204.383, 'NULL', 'NULL', 'NULL', 100, 'inf', 8.776, 0, 'NULL', 'NULL', 'NULL', 'NULL', 9.678, 0.21, 9.89, 3.43, 0),
'Pb204': ('Pb', '204', 'Lead', 204, 82, 203.973, -25.1097, '0', '+', 1.4, 'inf', 10.893, 0, 'NULL', 'NULL', 0, 0, 12.3, 0, 12.3, 0.65, 0.342601),
'Pb206': ('Pb', '206', 'Lead', 206, 82, 205.974, -23.7854, '0', '+', 24.1, 'inf', 9.221, 0, 'NULL', 'NULL', 0, 0, 10.68, 0, 10.68, 0.03, 0.0379272),
'Pb207': ('Pb', '207', 'Lead', 207, 82, 206.976, -22.4519, '0', '+', 22.1, 'inf', 9.286, 0, 'NULL', 'NULL', 0.14, 0, 10.82, 0.002, 10.82, 0.699, 0.0243158),
'Pb208': ('Pb', '208', 'Lead', 208, 82, 207.977, -21.7485, '0', '+', 52.4, 'inf', 9.494, 0, 'NULL', 'NULL', 0, 0, 11.34, 0, 11.34, 0.00048, 0.019883),
'Pb': ('Pb', 'nat', 'Lead', 207, 82, 207.2, 'NULL', 'NULL', 'NULL', 100, 'inf', 9.401, 0, 'NULL', 'NULL', 'NULL', 'NULL', 11.115, 0.003, 11.118, 0.171, 0),
'Bi': ('Bi', 'nat', 'Bismuth', 209, 83, 208.98, -18.2585, '9/2', '-', 100, 'inf', 8.53, 0, 8.26, 8.74, 0.22, 0.22, 9.148, 0.0084, 9.156, 0.0338, 0),
'Po': ('Po', 'nat', 'Polonium', 209, 84, 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 0),
'At': ('At', 'nat', 'Astatine', 210, 85, 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 0),
'Rn': ('Rn', 'nat', 'Radon', 222, 86, 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 0),
'Fr': ('Fr', 'nat', 'Francium', 223, 87, 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 0),
'Ra': ('Ra', 'nat', 'Radium', 226, 88, 'NULL', 'NULL', '0', '+', 100, '1600y', 10, 0, 'NULL', 'NULL', 0, 0, 13, 0, 13, 12.8, 0),
'Ac': ('Ac', 'nat', 'Actinium', 227, 89, 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 0),
'Th': ('Th', 'nat', 'Thorium', 232, 90, 232.038, 35.4483, '0', '+', 100, 'inf', 10.31, 0, 'NULL', 'NULL', 0, 0, 13.36, 0, 13.36, 7.37, 0),
'Pa': ('Pa', 'nat', 'Protactinium', 231, 91, 231.036, 'NULL', '3/2', '-', 100, '159200y', 9.1, 0, 'NULL', 'NULL', 'NULL', 'NULL', 10.4, 0.1, 10.5, 200.6, 0),
'U233': ('U', '233', 'Uranium', 233, 92, 233.04, 36.92, '5/2', '+', 'NULL', '159200y', 10.1, 0, 'NULL', 'NULL', -1, 0, 12.8, 0.1, 12.9, 574.7, 0.439886),
'U234': ('U', '234', 'Uranium', 234, 92, 234.041, 38.1466, '0', '+', 0.0054, 'inf', 12.4, 0, 'NULL', 'NULL', 0, 0, 19.3, 0, 19.3, 100.1, 1.17034),
'U235': ('U', '235', 'Uranium', 235, 92, 235.044, 40.9205, '7/2', '-', 0.7204, 'inf', 10.5, 0, 'NULL', 'NULL', -1.3, 0, 13.78, 0.2, 14, 680.9, 0.556195),
'U238': ('U', '238', 'Uranium', 238, 92, 238.051, 47.3089, '0', '+', 99.2742, 'inf', 0, 0, 'NULL', 'NULL', 0, 0, 8.871, 0, 8.871, 2.68, 1),
'U': ('U', 'nat', 'Uranium', 238, 92, 238.029, 'NULL', 'NULL', 'NULL', 100, 'inf', 8.417, 0, 'NULL', 'NULL', 'NULL', 'NULL', 8.903, 0.005, 8.908, 7.57, 0),
'Np': ('Np', 'nat', 'Neptunium', 237, 93, 'NULL', 44.8733, '5/2', '+', 100, '2144000y', 10.55, 0, 'NULL', 'NULL', 'NULL', 'NULL', 14, 0.5, 14.5, 175.9, 0),
'Pu238': ('Pu', '238', 'Plutonium', 238, 94, 238.05, 46.1647, '0', '+', 'NULL', '87.74y', 14.1, 0, 'NULL', 'NULL', 0, 0, 25, 0, 25, 558, 0),
'Pu239': ('Pu', '239', 'Plutonium', 239, 94, 239.052, 48.5899, '1/2', 'NULL', 'NULL', '24110y', 7.7, 0, 'NULL', 'NULL', -1.3, 0, 7.5, 0.2, 7.7, 1017.3, 0),
'Pu240': ('Pu', '240', 'Plutonium', 240, 94, 240.054, 50.127, '0', '+', 'NULL', '6561y', 3.5, 0, 'NULL', 'NULL', 0, 0, 1.54, 0, 1.54, 289.6, 0),
'Pu242': ('Pu', '242', 'Plutonium', 242, 94, 242.059, 54.7184, '0', '+', 'NULL', '375000y', 8.1, 0, 'NULL', 'NULL', 0, 0, 8.2, 0, 8.2, 18.5, 0),
'Pu': ('Pu', 'nat', 'Plutonium', 244, 94, 'NULL', 'NULL', 'NULL', 'NULL', 100, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 0),
'Am': ('Am', 'nat', 'Americium', 243, 95, 'NULL', 'NULL', '5/2', '-', 100, '7370y', 8.3, 0, 'NULL', 'NULL', -2, 0, 8.7, 0.3, 9, 75.3, 0),
'Cm244': ('Cm', '244', 'Curium', 244, 96, 244.063, 58.4537, '0', '+', 'NULL', '18.1y', 9.5, 0, 'NULL', 'NULL', 0, 0, 11.3, 0, 11.3, 16.2, 0),
'Cm246': ('Cm', '246', 'Curium', 246, 96, 246.067, 62.6184, '0', '+', 'NULL', '4706y', 9.3, 0, 'NULL', 'NULL', 0, 0, 10.9, 0, 10.9, 1.36, 0),
'Cm248': ('Cm', '248', 'Curium', 248, 96, 248.072, 67.3922, '0', '+', 'NULL', '348000y', 7.7, 0, 'NULL', 'NULL', 0, 0, 7.5, 0, 7.5, 3, 0),
'Cm': ('Cm', 'nat', 'Curium', 247, 96, 'NULL', 'NULL', 'NULL', 'NULL', 100, 'inf', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 0) }


###############################################################################
###############################################################################
###############################################################################
###############################################################################
