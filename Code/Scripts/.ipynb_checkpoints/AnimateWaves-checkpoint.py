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
  Animate result of 2D finite difference simulation
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

def animateacousticwaves(inpfile,fps,clipamplitude=0.1,v_inc = 20,showanim=False, outfile = "acoustic_animation.mp4") :
    """
     Function to 'animate' the results of a 2D acoustic finite difference simulation. It produces (and saves) an .mp4 movie.

    Args:
       inpfile (string): input HDF5 file name
       fps (int): frames per second
       clipamplitude (float): amplitude clipping factor
       v_inc (int): increment for quiver plot of v
       showanim (bool): show plot or not
       outfile (str): name of file to store
    
    """
    ##============================
    every = 1 
    kind = "acoustic"
    fl = inpfile

    ##=========================================

    h=h5.File(fl,"r")
    press = h["press"][:,:,:].copy()
    srctf = h["sourcetf"][:,:].copy()
    dt = h["dt"][()]
    dh = h["dh"][()]
    nx = h["nx"][()]
    ny = h["ny"][()]
    DeltaT = h["DeltaT"][:,:]
    vx = h["vx"][:,:]
    vy = h["vy"][:,:]
    snapevery = h["snapevery"][()]
    comp_fact = h["comp_fact"][()]
    #v_inc = h["v_inc"][()]
    ijrec = h["recij"][:,:].copy()
    h.close() 
    
    # Pick an STF for demonstration
    srctf = srctf[0,:]
    ##################

    N=press.shape[2]
    pmax = clipamplitude*abs(press).max()
    pmin = -pmax

    # colormap for wavefield
    cmap = plt.cm.gray_r #hot_r #gray_r #jet #RdGy_r #viridis

    # time axis
    nt = srctf.size
    t = np.arange(0.0,dt*nt,dt)
    
    # Compress model on save grid #
    save_nx = press.shape[0]
    save_ny = press.shape[1]
    
    # define coordinates
    x = np.linspace(0,DeltaT.shape[0]*dh, save_nx)
    y = np.linspace(0,DeltaT.shape[1]*dh, save_nx)
    
    # Average model on save grid:
    DeltaT = DeltaT.reshape(save_nx, comp_fact, save_ny, comp_fact).mean(axis = (1,3))
    vx = vx.reshape(save_nx, comp_fact, save_ny, comp_fact).mean(axis = (1,3))
    vy = vy.reshape(save_nx, comp_fact, save_ny, comp_fact).mean(axis = (1,3))
    
    # put receiver in perspective
    ijrec[0,:] = ijrec[0,:]//comp_fact
    ijrec[1,:] = ijrec[1,:]//comp_fact
    
    ####################################################################################

    fig1 = plt.figure(figsize=(12,5), constrained_layout=True) #constrained layout: elements are properly arranged and do not overlap

    gs = gridspec.GridSpec(1, 3, figure=fig1) # create 1x3-subplot
    gs.update(left=0.05,     # left margin of figure
              right=0.95,    # right margin of figure
              wspace=0.05,   # hor. spacing between subplots
              hspace=0.25)   # ver. spacing between subplots

    # Source Time Function
    sp1 = plt.subplot(gs[0, 0])  # first subplot location
    plt.title("Source Time Function")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    sep1 = plt.plot(t,srctf,'k')
    reddot, = plt.plot(0,srctf[0],'or')

    # Wavefield and Model
    sp2 = plt.subplot(gs[0, 1:]) # second subplot location
    plt.title("Wavefield, Clip Amplitude: {}".format(clipamplitude))
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    
    
    # DeltaT-Model
    DT = plt.pcolormesh(x,y,DeltaT[:,:].T,cmap=plt.cm.inferno)

    cbar1 = plt.colorbar(DT, aspect=50)
    cbar1.set_label("$\Delta T$")
    
    # v-Model
    plt.quiver(x[::v_inc],y[::v_inc],vx[::v_inc, ::v_inc].T, vy[::v_inc, ::v_inc].T, pivot = 'middle', width = 0.0015,color='lightsteelblue')
    
    # Pressure wave, partly transparent
    wav = plt.pcolormesh(x,y,press[:,:,1].T,vmin=pmin,vmax=pmax,cmap=cmap, animated=True, alpha=0.4)
    
    # Reciever positions
    plt.scatter(x[ijrec[0,:]],y[ijrec[1,:]],marker="v",color="k")

    
    plt.gca().set_aspect('equal')

    ######################################################

    def updatefig_acou(n):
        reddot.set_data([t[(n-1)*snapevery]],[srctf[(n-1)*snapevery]])
        wav.set_array(press[:,:,n].T)
        print(f'Frame {n*snapevery}', end='\r')
        return (wav,reddot)

    ######################################################
    # ***Animation iteration***
    ani = animation.FuncAnimation(fig1, updatefig_acou, frames=range(0,N,every), interval=1/fps*1000, blit=False)

    # Storing the animation
    mywriter = animation.FFMpegWriter(fps=fps)
    
    print('Saving the Video as ' + outfile)
    ani.save(outfile,dpi=96,writer=mywriter)


    ##################
    if showanim:
        plt.show()
        
    return ani
