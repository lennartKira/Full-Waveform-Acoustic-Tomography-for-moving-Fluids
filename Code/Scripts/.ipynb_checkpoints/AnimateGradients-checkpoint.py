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
  Animate the adjoint and forward fields, as well as the formation of the gradients
"""

#######################################################################
#######################################################################

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import gridspec

import h5py as h5

import numpy as np
import scipy.integrate as spi
import scipy.signal as sps

#######################################################################
# ######################################################################

def animategradients(inpfile,grad_index,fps,clipamplitude=0.1,v_inc = 5,showanim=False, outfile = "gradient_animation.mp4") :
    """
     Function to 'animate' the results of an inverse simulation of the 2D acoustic FWI. It produces (and saves to a file) an .mp4 movie.

    Args:
       inpfile (string): input HDF5 file name
       grad_index (int): which component of the gradient shall be plotted (0 -> c, 1 -> vx, 2 -> vy)
       fps (int): frames per second
       clipamplitude (float): amplitude clipping factor
       v_inc (int): increment for quiver plot of v
       showanim (bool): show plot or not
       outfile (str): name of file to store
    
    """
    ##============================
    every = 1
    fl=inpfile
    components = ["c", "v_x", "v_y"]

    ##=========================================

    h=h5.File(fl,"r")
    press = h["press"][:,:,:].copy()
    press_adj = h["press_adj"][:,:,:].copy()
    grad = h["grad_anim"][:,:,:,:].copy()
    srctf = h["srctf"][0,:].copy()
    dt = h["dt"][()]
    dh = h["dh"][()]
    nx = h["nx"][()]
    ny = h["ny"][()]
    snapevery = h["snapevery"][()]
    ijsrc = h["srcij"][:,:].copy()
    h.close()

    gradient = grad[grad_index, :, :]
    
    N=press.shape[2]
    pmax = clipamplitude*abs(press).max()
    pmin = -pmax
    pmax_adj = clipamplitude/4*abs(press_adj).max()
    pmin_adj = -pmax_adj
    gradmax = np.max(gradient[80:-80,80:-80,:]) # the colormap is oriented on the interior values of the gradients, not on singularities near sources
    gradmin = -gradmax
    cmap = plt.cm.gray_r #hot_r #gray_r #jet #RdGy_r #viridis
    cmap_adj = plt.cm.RdGy
    cmap_grad = plt.cm.seismic

    ######################################################

    def updatefig_acou(n):
        reddot.set_data([t[(n-1)*snapevery]],[srctf[(n-1)*snapevery]])
        grad_evo.set_array(gradient[:,:,n].T)
        wav.set_array(press[:,:,n].T)
        wav_adj.set_array(press_adj[:,:,n].T)
        print(f'Frame {n*snapevery}', end='\r')
        return (reddot,grad_evo,wav,wav_adj)

    ######################################################

    nt = srctf.size          # obtaining the number of time steps from the stf
    t = np.arange(0.0,dt*nt,dt)

    # Compress on save grid###########################################################
    save_nx = press.shape[0]
    save_ny = press.shape[1]
    
    ratio_nx = nx//save_nx
    ratio_ny = ny//save_ny
    
    # define coordinates
    x = np.linspace(0,press.shape[0]*dh*ratio_nx, save_nx)
    y = np.linspace(0,press.shape[1]*dh*ratio_ny, save_nx)
    
    # put receiver in perspective
    ijsrc[0,:] = ijsrc[0,:]//ratio_nx
    ijsrc[1,:] = ijsrc[1,:]//ratio_ny
    
    ####################################################################################

    fig1 = plt.figure(figsize=(12,5), constrained_layout=True) #constrained layout: elements are properly arranged and do not overlap

    gs = gridspec.GridSpec(1, 3, figure=fig1)                  # create 1x4-subplot
    gs.update(left=0.05,     # left margin of figure
              right=0.95,    # right margin of figure
              wspace=0.05,   # hor. spacing between subplots
              hspace=0.25)   # ver. spacing between subplots

    # Source Time Function
    sp1 = plt.subplot(gs[0, 0])  # first subplot location
    plt.title("Adjoint Source")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    sep1 = plt.plot(t,srctf,'k')
    reddot, = plt.plot(0,srctf[0],'or')

    # Wavefield and Model
    sp2 = plt.subplot(gs[0, 1:]) # second subplot location
    plt.title(f"Wavefields and Gradient w.r.t. {components[grad_index]}")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    
    
    # Gradient evolution
    grad_evo = plt.pcolormesh(x,y,gradient[:,:,1].T,
                              vmin=gradmin,vmax=gradmax,cmap=cmap_grad, animated=True)
    # Pressure wave (starts with first timestep)
    wav = plt.pcolormesh(x,y,-press[:,:,1].T,vmin=pmin,vmax=pmax,cmap=cmap, animated=True, alpha=0.3)
    # Adjoint wave (same clip as pressure)
    wav_adj = plt.pcolormesh(x,y,press_adj[:,:,1].T,vmin=pmin_adj,vmax=pmax_adj,cmap=cmap_adj, animated=True, alpha=0.3)
    
    # Reciever positions
    plt.scatter(x[ijsrc[0,:]],y[ijsrc[1,:]],marker="*",color="grey")

    
    plt.gca().set_aspect('equal')
    # ***Animation iteration***
    ani = animation.FuncAnimation(fig1, updatefig_acou, frames=range(0,N,every), interval=150, blit=False)

    # Storing the animation
    mywriter = animation.FFMpegWriter(fps = fps)
    ani.save(outfile,dpi=96,writer=mywriter)


    ##################
    if showanim:
        plt.show()
        
    return ani
