# -----------------------------------------------------------------------------
# This routine is part of the publication "Full-Waveform Acoustic Tomography for Fluid Temperature and Velocity" (Kira & Noir, 2025) 
# and developed to carry out synthetic Full-Waveform Inversions to scan laboratory scale flows.
# Copyright (C) 2025 Lennart Kira
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

"""
A collection of source time functions. 
"""

#############################################################

import numpy as np
import scipy.special

#############################################################

def gaussource( t, t0, f0 ):
    """
    Derivative of a Gaussian source time function.

    Args: 
      t (ndarray): time array
      t0 (float): time shift for the onset 
      f0 (float): central frequency

    Returns:
      source (ndarray): the source time function

    """
    # boh = f0 .* (t-t0)
    # source = -8.0*boh.*exp( -boh.^2/(4.0*f0)^2 )
    a = (np.pi*f0)**2
    #source = - 8.0*a*(t-t0)*np.exp( -a*(t-t0)**2 )
    source = 8.0*a*np.exp( -a*(t-t0)**2 ) ## Test
    
    return source

#############################################################

def rickersource( t, t0, f0 ):
    """
    Ricker wavelet source time function.
    
    Args: 
      t (ndarray): time array
      t0 (float): time shift for the onset 
      f0 (float): central frequency

    Returns:
      source (ndarray): the source time function

    """
    b = (np.pi*f0*(t-t0))**2
    w = (1.0-2.0*b)*np.exp(-b)
    return w

#############################################################

def ricker_1st_derivative_source( t, t0, f0 ):
    """
    First derivative of a wavelet source time function.
    
    Args: 
      t (ndarray): time array
      t0 (float): time shift for the onset 
      f0 (float): central frequency

    Returns:
      source (ndarray): the source time function
    """
    source = 2 * np.pi**2 * (t - t0) * f0**2 * np.exp(- np.pi**2 * (t - t0)**2 * f0**2) * (2 * np.pi**2 * (t - t0)**2 * f0**2 - 3)
    return source

#############################################################

def cossource(t,t0,f0,width):
    """
    Cosine wavelet source time function - modulated by a two-sided erfc.
    
    Args: 
      t (ndarray): time array
      t0 (float): time shift for the onset 
      f0 (float): central frequency
      width (float): width/duration of the signal

    Returns:
      source (ndarray): the source time function
      envelope (ndarray): envelope of STF
    """
    
    envelope = 1/2*np.where(t <= t0, scipy.special.erfc(2*np.pi/width*(t0-(t+width/2))), scipy.special.erfc(2*np.pi/width*((t-width/2)-t0)))
    
    source = np.cos(2 * np.pi*f0*(t-t0))*envelope
    
    return source, envelope
