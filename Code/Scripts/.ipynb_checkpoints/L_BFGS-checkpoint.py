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
Functions to perform an N-dimensional gradient descent with the L-BFGS Algorithm
"""

import numpy as np

def compute_stepdir(grad, s, y, gamma):
    
    """
    Compute the step direction for a given current gradient and information on former step directions and gradients.
    See Nocedal and Wright (2006) for details.
    
    Args:
        grad (ndarray(nx,ny)): gradient of the misfit for the current model (grad_k)
        s (ndarray(nx,ny,m)): the most recent m steps in the model space (m_k - m_(k-1), m_(k-1) - m_(k-2), ... , m_(k-m+1) - m_(k-m))
        y (ndarray(nx,ny,m)): the most recent m changes in the gradients (grad_k - grad_(k-1), ... , grad_(k-m+1) - grad_(k-m))
        gamma (float): Approximate residual of an eigenvalue of the current Hessian 

    Returns:
        r (ndarray(nx,ny)): step direction for next model update (-H_k*grad_k)
        
    """
    
    m = np.shape(s)[2]
    a = np.zeros(m)
    rho = np.zeros(m)
    q = grad
    
    for i in reversed(range(m)):
        rho[i] = 1/np.sum(y[:,:,i]*s[:,:,i])
        a[i] = rho[i]*np.sum(s[:,:,i]*q)
        q = q - a[i]*y[:,:,i]
        
    r = gamma*q
    
    for i in range(m):
        b = rho[i]*np.sum(y[:,:,i]*r)
        r = r + s[:,:,i]*(a[i] - b)
        
    r = r*(-1)
    
    return r