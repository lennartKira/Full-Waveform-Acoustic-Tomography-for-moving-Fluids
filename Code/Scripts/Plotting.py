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
from matplotlib import gridspec
import matplotlib.ticker as ticker

'''
Module to hide superfluously large plotting routines.
'''

### Plot the complete model for DT and v
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def PlotModel(x,y,DeltaT,vx,vy, srcpos = None, recpos = None, v_inc=5, comp_fact = None, outfile = None):
    '''
    Plots model with pcolormesh of DT (inferno) and a quiver-plot of v on the left as well as a pcolormesh the two flow components on the right (viridis).
    Input: Grid (x,y), Temperature Anomaly (DeltaT), Flow Field (vx, vy)
    Optional Input: Source- and Receiver-Positions (srcpos, recpos), increment for quiver plot (v_inc), compression factor to reduce grid resolution for faster plotting (comp_fact)
    '''
    
    if comp_fact is not None:
        comp_nx = DeltaT.shape[0]//comp_fact
        comp_ny = DeltaT.shape[1]//comp_fact

        # redefine coordinates
        x = np.linspace(0,x.max(), comp_nx)
        y = np.linspace(0,y.max(), comp_ny)

        DeltaT = DeltaT.reshape(comp_nx, comp_fact, comp_ny, comp_fact).mean(axis = (1,3))
        vx = vx.reshape(comp_nx, comp_fact, comp_ny, comp_fact).mean(axis = (1,3))
        vy = vy.reshape(comp_nx, comp_fact, comp_ny, comp_fact).mean(axis = (1,3))
        
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 2)
    
    gs.update(wspace = 0,
             hspace = 0.25)
    
    ax1 = plt.subplot(gs[:, 0])  # first column
    ax2 = plt.subplot(gs[0, 1])  # second column, first row
    ax3 = plt.subplot(gs[1, 1])  # second column, second row
    
    ## Vectorfield in Front of Colormap
    im1 = ax1.pcolormesh(x,y,DeltaT.T, cmap="inferno")
    if srcpos is not None:
        ax1.plot(srcpos[:,0], srcpos[:,1], "*w", markersize = 15, label = 'Sources')
    if recpos is not None:
        ax1.plot(recpos[:,0], recpos[:,1], "vc", label = 'Receivers')
    
    # Select every ... grid point for v representation
    ax1.quiver(x[::v_inc],y[::v_inc],vx[::v_inc, ::v_inc].T, vy[::v_inc, ::v_inc].T, pivot = 'middle', color = 'snow')
    
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    cbar1 = fig.colorbar(im1, ax = ax1, fraction=0.08*(10/16), pad=0.04, label = '$\Delta T$')
    ax1.set_aspect('equal')
    
    ## vx-Colormap
    im2 = ax2.pcolormesh(x,y,vx.T, cmap="viridis")
    cbar2 = fig.colorbar(im2, ax = ax2, fraction=0.046*(10/16), pad=0.04, label = '$vx$')
    ax2.set_ylabel("y")
    ax2.set_aspect('equal')
    
    ## vy-Colormap
    im3 = ax3.pcolormesh(x,y,vy.T, cmap="viridis")
    cbar3 = fig.colorbar(im3, ax = ax3, fraction=0.046*(10/16), pad=0.04, label = '$vy$')
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_aspect('equal')
    #***************************************************************
    if outfile:
        plt.savefig(otufile)
    plt.show()

    return

### Plotting for Model Evaluation:
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def PlotTemperature_SRvFWI(x, y, DT_true, DT_FWI, DT_SR, srcpos = None, RBF_centers = None, width = 13, height = 4):
    '''
    Plots True Temperature (left) to compare to FWI-reconstruction (middle) and Straight-Ray (SR) reconstruction (right)
    Input: SR-grid (x,y), temperature anomaly of FWI - avg. in SR-grid - and SR (DT_FWI, DT_SR)
    Optional Input: Positions of Radial Basis Function-Centers (RBF_centers), width and height of plot (width, height), position of sources (srcpos)
    '''

    
    fig = plt.figure(figsize = (width,height))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.06])
    
    gs.update(wspace = 0.2,
             hspace = 0.2)
    
    ax1 = plt.subplot(gs[0, 0])  # first column, first row
    ax2 = plt.subplot(gs[0, 1])  # second column, first row
    ax3 = plt.subplot(gs[0, 2])

    # get cbar limits from true model
    clim = [np.min([DT_true]), np.max([DT_true])]

    im1 = ax1.pcolormesh(x,y,DT_true.T, cmap="inferno", clim = clim)
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_title('True')
    ax1.set_aspect('equal')
    
    im2 = ax2.pcolormesh(x,y,DT_FWI.T, cmap="inferno", clim = clim)
    ax2.set_xlabel('x [m]')
    ax2.set_title('FWI')
    ax2.set_yticks([])
    ax2.set_aspect('equal')

    
    im3 = ax3.pcolormesh(x,y,DT_SR[:,:].T, cmap="inferno", clim = clim)
    if RBF_centers is not None:
        ax3.scatter(RBF_centers[:,0], RBF_centers[:,1], marker = 'o', c = "white")
    ax3.set_xlabel('x [m]')
    ax3.set_title('SRI')
    ax3.set_yticks([])
    ax3.set_aspect('equal')
    plt.gca().set_aspect('equal')

    if srcpos is not None:
        ax1.scatter(srcpos[:,0], srcpos[:,1], c = "tomato", marker = "x", s = 75, linewidths = 4)
        ax1.set_xlim(x.min(), x.max())
        ax1.set_ylim(y.min(), y.max())
        
        for i in range(srcpos.shape[0]):
            for j in range(srcpos.shape[0]):
                ax1.plot([srcpos[i,0], srcpos[j,0]], [srcpos[i,1], srcpos[j,1]], color = "azure", linestyle = "--", zorder = 1, linewidth = 1.5, alpha = 0.3)
    
    # Colorbar
    cbar_ax = plt.subplot(gs[0, 3])  # Dedicated axis for the colorbar
    cbar = fig.colorbar(im1, cax=cbar_ax, label='$\Delta T$ [K]')

    
    plt.show()
    
    return


def PlotVelocity_SRvFWI(x, y, vx_true, vy_true, vx_FWI, vy_FWI, vx_SR, vy_SR, v_inc = 1, width = 13, height = 4, srcpos = None, outfile = None):
    '''
    Plots True Flow Field (left) to compare to FWI-reconstruction (middle) and Straight-Ray (SR) reconstruction (right)
    Input: SR-grid (x,y), flow field of FWI (vx_FWI, vy_FWI) - avg. in SR-grid - and SR (vx_SR, vy_SR)
    Optional Input: Increment for quiver plot (v_inc), Positions of Radial Basis Function-Centers (RBF_centers), width and height of plot (width, height), position of sources (srcpos)
    '''
    v_true = np.sqrt(vx_true**2 + vy_true**2)
    v_FWI = np.sqrt(vx_FWI**2 + vy_FWI**2)
    v_SR = np.sqrt(vx_SR**2 + vy_SR**2)

    quiver_width = 5e-3
    quiver_scale = 1.2
    
    fig = plt.figure(figsize = (width,height))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.06])
    
    gs.update(wspace = 0.1,
             hspace = 0.2)
    
    ax1 = plt.subplot(gs[0, 0])  # first column, first row
    ax2 = plt.subplot(gs[0, 1])  # second column, first row
    ax3 = plt.subplot(gs[0, 2])  # first column second row
    
    clim = [np.min([v_true]), np.max([v_true])] #, v_FWI_avg, v_SR
    im1 = ax1.pcolormesh(x,y,v_true.T, cmap="viridis", clim = clim)
    ax1.quiver(x[::v_inc],y[::v_inc],vx_true[::v_inc,::v_inc].T,vy_true[::v_inc,::v_inc].T, color = "snow", width = quiver_width, scale = quiver_scale)
    ax1.set_ylabel('$y$ [m]')
    ax1.set_xlabel('$x$ [m]')
    ax1.set_title('True')
    ax1.set_aspect('equal')
    
    im2 = ax2.pcolormesh(x,y,v_FWI.T, cmap="viridis", clim = clim)
    ax2.quiver(x[::v_inc],y[::v_inc],vx_FWI[::v_inc,::v_inc].T,vy_FWI[::v_inc,::v_inc].T, color = "snow", width = quiver_width, scale = quiver_scale)
    ax2.set_xlabel('$x$ [m]')
    ax2.set_title('FWI')
    ax2.set_yticks([])
    ax2.set_aspect('equal')
    
    im3 = ax3.pcolormesh(x,y,v_SR[:,:].T, cmap="viridis", clim = clim)
    ax3.quiver(x[::v_inc],y[::v_inc],vx_SR[::v_inc,::v_inc].T,vy_SR[::v_inc,::v_inc].T, color = "snow", width = quiver_width, scale = quiver_scale)
    ax3.set_xlabel('$x$ [m]')
    ax3.set_title('Straight Ray')
    ax3.set_aspect('equal')
    ax3.set_yticks([])
    plt.gca().set_aspect('equal')
    
    # Colorbar
    cbar_ax = plt.subplot(gs[0, 3])  # Dedicated axis for the colorbar
    cbar = fig.colorbar(im3, cax=cbar_ax, label='$u$ [m/s]')

    if srcpos is not None:
        ax1.scatter(srcpos[:,0], srcpos[:,1], c = "white", marker = "x", s = 100, linewidths = 4)
        ax1.set_xlim(x.min(), x.max())
        ax1.set_ylim(y.min(), y.max()) 

        for i in range(srcpos.shape[0]):
            for j in range(srcpos.shape[0]):
                ax1.plot([srcpos[i,0], srcpos[j,0]], [srcpos[i,1], srcpos[j,1]], color = "azure", linestyle = "--", zorder = 1, linewidth = 1.5, alpha = 0.4)

    if outfile:
        plt.savefig(outfile, dpi = 300, bbox_inches = "tight")
    
    plt.show()
    
    return

### Plotting for Snapshots of Wavefields
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def PlotSnapshot_WaveField(x,y,DeltaT,vx,vy,press,comp_fact,ijrec,ijsrc,n_snapshot,clipamplitude,v_inc = 5):
    '''
    Plots model with pcolormesh of DT (inferno) and a quiver-plot of v as a background to the snapshot of a wavefield
    Input: Grid (x,y), Temperature Anomaly (DeltaT), Flow Field (vx,vy), snapshots of wavefield (press), compression factor to reduce grid resolution for faster plotting (comp_fact), receiver- and source-indices (ijrec,ijsrc), number of snapshot to plot (n_snapshot), amplitude at which wavefield shall be clipped (clipamplitude)
    Optional Input: Increment for quiver plot (v_inc)
    '''
    dh = x[1] - x[0]
    
    N=press.shape[2]
    pmax = clipamplitude*abs(press).max()
    pmin = -pmax
    # colormap for pressure field
    cmap = plt.cm.gray_r #hot_r #gray_r #jet #RdGy_r #viridis
    
    # Compress on save grid###
    save_nx = press.shape[0]
    save_ny = press.shape[1]
    
    # define coordinates
    x_comp = np.linspace(0,DeltaT.shape[0]*dh, save_nx)
    y_comp = np.linspace(0,DeltaT.shape[1]*dh, save_nx)
    
    # Average model on save grid:
    DeltaT_comp = DeltaT.reshape(save_nx, comp_fact, save_ny, comp_fact).mean(axis = (1,3))
    vx_comp = vx.reshape(save_nx, comp_fact, save_ny, comp_fact).mean(axis = (1,3))
    vy_comp = vy.reshape(save_nx, comp_fact, save_ny, comp_fact).mean(axis = (1,3))
    
    
    # put sources and receivers in perspective
    ijsrc_comp = np.zeros(ijsrc.shape, dtype=int)
    ijrec_comp = np.zeros(ijrec.shape, dtype=int)
    
    ijsrc_comp[0,:] = ijsrc[0,:]//comp_fact
    ijsrc_comp[1,:] = ijsrc[1,:]//comp_fact
    ijrec_comp[0,:] = ijrec[0,:]//comp_fact
    ijrec_comp[1,:] = ijrec[1,:]//comp_fact
    
    
    ###
    fig1 = plt.figure(figsize=(6,5), constrained_layout=True)
    
    gs = gridspec.GridSpec(1, 3, figure=fig1) # create 1x4-subplot
    
    gs.update(left=0.05,     # left margin of figure
              right=0.95,    # right margin of figure
              wspace=0.05,   # hor. spacing between subplots
              hspace=0.25)   # ver. spacing between subplots
    
    # Wavefield and Model
    
    # DeltaT-Model
    DT = plt.pcolormesh(x_comp,y_comp,DeltaT_comp[:,:].T,cmap=plt.cm.plasma)
    cbar1 = plt.colorbar(DT, fraction=0.07*(10/16))
    cbar1.set_label("$\Delta T$")
    
    # Pressure wave (starts with first timestep)
    wav = plt.pcolormesh(x_comp,y_comp,press[:,:,n_snapshot].T,vmin=pmin,vmax=pmax,cmap=cmap, animated=True, alpha=0.4)
    
    # v-Model
    plt.quiver(x_comp[::v_inc],y_comp[::v_inc],vx_comp[::v_inc, ::v_inc].T,vy_comp[::v_inc, ::v_inc].T, 
               pivot = 'middle', width = 0.0015,color='lightsteelblue')
    
    # Reciever positions
    plt.scatter(x_comp[ijsrc_comp[0,:]],y_comp[ijsrc_comp[1,:]],marker="*",color="white", s = 350)
    plt.scatter(x_comp[ijrec_comp[0,:]],y_comp[ijrec_comp[1,:]],marker="v",color="cyan")
    plt.xlabel("x")
    plt.ylabel("y")
    
    plt.gca().set_aspect('equal')
    #plt.savefig("../../Data/Snapshots/Snapshot_Libration_noDT_Sparse_SimAqu_t4000.png")
    plt.show()

    return



## Plotting Gradients
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def ModelvGradient(x,y,m,gradient,m_index,srcpos = None, recpos = None, comp_fact = 20, v_inc = 5,width = 10, height = 5):
    '''
    Plots model on the left and the gradient w.r.t. the specified model parameter to the right.
    Input: Coordiantes (x,y), the model (m), the gradients (gradient), the index  (m_index) of the model parameter to plot (0 -> c, 1 -> vx, 2 -> vy) 
    Optional Input: Source positions (srcpos), receiver positions (recpos), compression factor (comp_fact) for memory saving,
                    increment for quiver plot (v_inc), width and height of plot (width, height)
    '''

    components = ["c", "v_x", "v_y"]

    if comp_fact is not None:
        comp_nx = m.shape[1]//comp_fact
        comp_ny = m.shape[2]//comp_fact

        # redefine coordinates
        x = np.linspace(0,x.max(), comp_nx)
        y = np.linspace(0,y.max(), comp_ny)

        m = m.reshape(3,comp_nx, comp_fact, comp_ny, comp_fact).mean(axis = (2,4))
        gradient = gradient.reshape(3,comp_nx, comp_fact, comp_ny, comp_fact).mean(axis = (2,4))
    
    fig = plt.figure(figsize=[width, height])
    gs = gridspec.GridSpec(1, 2)
    
    gs.update(wspace = 0.35,
             hspace = 0.3)
    
    ax1 = plt.subplot(gs[0, 0])  # first column
    ax2 = plt.subplot(gs[0, 1])  # second column

    if m_index == 0:
        im1 = ax1.pcolormesh(x,y,m[m_index,:,:].T, cmap="inferno")
        cbar1 = fig.colorbar(im1,fraction=0.07*(10/16), pad=0.04,label = "$\\Delta T$")
    else:
        im1 = ax1.pcolormesh(x,y,np.sqrt(m[1,:,:]**2 + m[2,:,:]**2).T, cmap="viridis")
        ax1.quiver(x[::v_inc],y[::v_inc],m[1,::v_inc,::v_inc].T,m[2,::v_inc,::v_inc].T, pivot = 'middle', color = "lightsteelblue")
        cbar1 = fig.colorbar(im1,fraction=0.07*(10/16), pad=0.04,label = "$v$")
    
    ax1.set_ylabel('y')
    ax1.set_xlabel('x')
    ax1.set_aspect('equal')
    
    clim = [-np.max(np.abs(gradient[m_index,:,:])), np.max(np.abs(gradient[m_index,:,:]))]
    im2 = ax2.pcolormesh(x,y,gradient[m_index,:,:].T, cmap="seismic", clim = clim)
    cbar2 = fig.colorbar(im2,fraction=0.07*(10/16), pad=0.04,label = f'$\\partial \\chi / \\partial {components[m_index]}$', format = "%g")
    ax2.set_xlabel('x')
    ax2.set_aspect('equal')

    if srcpos is not None:
        ax2.scatter(srcpos[:,0], srcpos[:,1], c = "lightcyan", marker = "*", s = 100, linewidths = 4)
        
    if srcpos is not None:
        ax2.scatter(recpos[:,0], recpos[:,1], c = "lightcyan", marker = "v", s = 100, linewidths = 4)

    ax2.set_xlim(x.min(), x.max())
    ax2.set_ylim(y.min(), y.max())
    plt.show()

    return


def PlotReciprocal_Grad(x,y,gradient_ab,gradient_ba,gradient,m_index,width = 10, height = 6):
    '''
    Plots Gradients for shot a -> b and b -> a on the left and the sum of the two reciprocal gradients on the right.
    Input: Grid (x,y), gradient for shot a -> b (gradient_ba) and b -> a  (gradient_ba), the sum of the two (gradient), 
           the index (m_index) of the model parameter (0 -> c, 1 -> vx, 2 -> vy)
    Optional Input: width and height of plot (width, height)
    '''

    components = ["c","v_x", "v_y"]
    
    # display gradient:
    fig = plt.figure(figsize=[width, height])
    gs = gridspec.GridSpec(2, 2)
    
    gs.update(wspace = 0.4,
             hspace = 0.3)
    
    ax1 = plt.subplot(gs[0, 0])  # first column
    ax2 = plt.subplot(gs[1, 0])  # second column, first row
    ax3 = plt.subplot(gs[:, 1])  # second column, second row
    
    clim = [-np.max(np.abs(gradient[m_index,:,:])), np.max(np.abs(gradient[m_index,:,:]))]
    im1 = ax1.pcolormesh(x,y,gradient_ab[m_index,:,:].T, cmap="seismic", clim = clim)
    cbar1 = fig.colorbar(im1,label = f'$\\partial \\chi / \\partial {components[m_index]}$')
    cbar1.set_ticks([clim[0], 0, clim[1]])
    cbar1.set_ticklabels(["%.1e" %clim[0], '0', "%.1e" %clim[1]])
    ax1.set_ylabel('y')
    #ax1.set_title('a $\\rightarrow$ b')
    ax1.set_aspect('equal')
    
    im2 = ax2.pcolormesh(x,y,gradient_ba[m_index,:,:].T, cmap="seismic", clim = clim)
    cbar2 = fig.colorbar(im2,label = f'$\\partial \\chi / \\partial {components[m_index]}$')
    cbar2.set_ticks([clim[0], 0, clim[1]])
    cbar2.set_ticklabels(["%.1e" %clim[0], '0', "%.1e" %clim[1]])
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    #ax2.set_title('b $\\rightarrow$ a')
    ax2.set_aspect('equal')
    
    clim_sum = [-np.max(gradient[m_index,:,:]), np.max(gradient[m_index,:,:])]
    im3 = ax3.pcolormesh(x,y,gradient[m_index,:,:].T,cmap="seismic", clim = clim_sum)
    cbar3 = fig.colorbar(im3,fraction=0.08*(10/16), pad=0.04,label = f'$\\partial \\chi / \\partial {components[m_index]}$')
    cbar3.set_ticks([clim_sum[0], 0, clim_sum[1]])
    cbar3.set_ticklabels(["%.1e" %clim_sum[0], '0', "%.1e" %clim_sum[1]])
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    #ax3.set_title('Sum of filtered Gradients')
    ax3.set_aspect('equal')

    plt.show()
    return