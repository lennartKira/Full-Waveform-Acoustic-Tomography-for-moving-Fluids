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

import numpy as np
import matplotlib.pyplot as plt


def Transducers4x4(nx,ny):
    ## Rectangular array:
    ijsrc = np.array([[nx//5, 1],   # lower boundary
                      [2*nx//5, 1],
                      [3*nx//5, 1],
                      [4*nx//5, 1],
                      [nx//5, ny-2],  # upper boundary
                      [2*nx//5, ny-2],
                      [3*nx//5, ny-2],
                      [4*nx//5, ny-2],
                      [1, ny//5],   # left boundary
                      [1, 2*ny//5],
                      [1, 3*ny//5],
                      [1, 4*ny//5],
                      [nx-2, ny//5],  # right boundary
                      [nx-2, 2*ny//5],
                      [nx-2, 3*ny//5],
                      [nx-2, 4*ny//5]]).T

    ijrec = ijsrc
    return ijsrc, ijrec
    
def Transducers4x2(nx,ny):
    # Rectangular array - Sparse:
    ijsrc = np.array([[nx//3, 1],   # lower boundary
                      [2*nx//3, 1],
                      [nx//3, ny-2],  # upper boundary
                      [2*nx//3, ny-2],
                      [1, ny//3],   # left boundary
                      [1, 2*ny//3],
                      [nx-2, ny//3],  # right boundary
                      [nx-2, 2*ny//3]]).T

    ijrec = ijsrc
    return ijsrc, ijrec
    
def Transducers2x2(nx,ny):
    ## Very Sparse Array - 2x2:
    ijsrc = np.array([[nx//5, 1],   # lower boundary
                      [3*nx//5, 1],
                      [1, ny//5], # left boundary
                      [1, 3*ny//5]]).T

    ijrec = ijsrc
    return ijsrc, ijrec

def plot(axes,Transducers,nx,ny,dh):
    sources = Transducers[0]*dh
    receivers = Transducers[1]*dh

    for i in range(sources.shape[1]):
        for j in range(receivers.shape[1]):
            axes.plot([sources[0,i], receivers[0,j]], [sources[1,i], receivers[1,j]], color = "darkblue", zorder = 1)

    axes.scatter(sources[0, :], sources[1,:], color = "lightgreen", marker = "*", s = 100, zorder = 2)
    axes.set_xlim((0,nx*dh))
    axes.set_ylim((0,ny*dh))
    axes.set_aspect('equal')

    return
                


