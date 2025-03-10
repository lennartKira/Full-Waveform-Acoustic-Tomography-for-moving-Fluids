##################################################################################

# Importing packages
#################
import numpy as np
#################

'''
Constituive Relationship of the acoustic wavespeed and temperature in an ideal gas

'''

def c2(DeltaT, T0, DeltaT0):
    '''
    Dimensionless function returning the square of the sound speed for a given temperature field.
    
    Args: 
      DeltaT (ndarray): Temperature Field (Dimensionless)
      T0 (float): Background Temperature
      DeltaT0 (float): Typical Temperature Anomaly  

    Returns:
      c2 (ndarray): square of sound speed (Dimensionless)
    '''
    
    c_sq = 1. + DeltaT0/T0*DeltaT
    
    return c_sq

def c_inv(c, T0, DeltaT0):
    '''
    Inverse function of c2.
    
    Args: 
      c (ndarray): Sound Speed (Dimensionless)
      T0 (float): Background Temperature
      DeltaT0 (float): Typical Temperature Anomaly  

    Returns:
      DeltaT (ndarray): Temperature Field (Dimensionless)
    '''
    
    DeltaT = (c**2. - 1)*T0/DeltaT0
    
    return DeltaT




