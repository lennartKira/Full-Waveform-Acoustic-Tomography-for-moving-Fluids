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
Functions to calculate the adjoint and time reversed wavefield. Additionally, some post processing tools for the gradients of the misfit.
"""

#######################################################################
#######################################################################

import numpy as np
import sys
import h5py as h5
import torch
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

# Gaussian filter in wavenumber domain
def Gaussfilter(kx_vals, ky_vals, k_max):
    kx_vals_2d, ky_vals_2d = np.meshgrid(kx_vals, ky_vals)
    f = 1 / (2 * np.pi) * np.exp(-(kx_vals_2d**2 + ky_vals_2d**2) / (2 * k_max**2))
    return f

#############################################################################
#############################################################################

def adjointsource_L2(p0, seism):
    
    '''
    Computes the adjoint source based on the L2-Misfit between observed wavefield 'p0' and estimated wavefield 'seism'.
    Inputs:
    p0   : shape = (N_rec, nt) - observed wavefield
    seism: shape = (N_rec, nt) - estimated wavefield with current model
    '''
    
    adjsrc= (p0-seism)
    
    return adjsrc

def filter_SepGaussian(grad, dh, lambda_min):
    '''
    Filters the gradients (or any nx X ny-array) with a gaussian filter of width 'lambda_min' by applzing two 1D filters
    grad (ndarray(nx,ny)): gradient of the misfit w.r.t. a specific model parameter (c,vx or vy)
    dh (float): grid step
    lambda_min (float): standard deviation of the Gaussian filter with which 'grad' will be smoothed
    '''

    temp = gaussian_filter1d(grad, lambda_min//dh, axis = 0)
    filtered_grad = gaussian_filter1d(temp, lambda_min//dh, axis = 1)

    return filtered_grad

def filter_grad(grad, dh, lambda_min, padding = False, pad_width = None, detrend = True):

    '''
    Filters the gradients (or any nx X ny-array) with a gaussian filter of width 'lambda_min' using an FFT.
    Inputs:
    grad (ndarray(nx,ny)): gradient of the misfit w.r.t. a specific model parameter (c,vx or vy)
    dh (float): grid step
    lambda_min (float): standard deviation of the Gaussian filter with which 'grad' will be smoothed
    padding (bool): wether the boundary of the gradients should be padded before transformed
    pad_width (int): number of grid points to pad the boundaries of 'grad' for the FFT
    detrend (bool): wether the gradients should be detrended before transformed
    '''
    
    if detrend:
        #define plane function (linear trend)
        def plane(coords, a, b, c):
            x,y = coords
            return a*x + b*y + c
        
        # obtain grid to perform plane fit on
        x, y = np.meshgrid(np.arange(grad.shape[0]), np.arange(grad.shape[1]))
        # flatten evth to use scipy's curve_fit
        coords = np.vstack((x.ravel(), y.ravel()))
        grad_flat = grad.ravel()

        # fit the linear trend
        plane_opt, _ = curve_fit(plane, coords, grad_flat)
        trend = plane(coords, *plane_opt).reshape(grad.shape[0],grad.shape[1])

        grad = grad - trend
    
    if padding:
        grad = np.pad(grad, pad_width = pad_width, mode = 'constant', constant_values = 0)
        
        
    
    # Compute Fourier Transform
    fft_grad = np.fft.fft2(grad)

    # Shift the zero frequency component to the center
    fft_grad_shifted = np.fft.fftshift(fft_grad)
    
    # Frequency values
    Nx = grad.shape[0]
    Ny = grad.shape[1]
    kx_vals = np.fft.fftshift(np.fft.fftfreq(Nx, dh)) * 2 * np.pi
    ky_vals = np.fft.fftshift(np.fft.fftfreq(Ny, dh)) * 2 * np.pi
    
    k_max = 1/lambda_min # as in 2D FT of Gaussian Kernel
    
    filt = Gaussfilter(ky_vals, kx_vals, k_max)

    filtered_fft_grad = fft_grad_shifted*filt

    # Reconstruct the array using inverse Fourier transform
    reconstructed_grad = np.fft.ifft2(np.fft.ifftshift(filtered_fft_grad)).real

    if padding:
        reconstructed_grad = reconstructed_grad[pad_width:-pad_width, pad_width:-pad_width]

    if detrend:
        print(reconstructed_grad.shape)
        print(grad.shape)
        reconstructed_grad += trend
        
    
    return reconstructed_grad

def filter_grad_zeropadding(grad, dh, lambda_min, pad_width):

    '''
    Filters the gradients (or any nx X ny-array) with a gaussian filter of width 'lambda_min' using an FFT and zero padding.
    Inputs:
    grad (ndarray(nx,ny)): Gradient of the misfit w.r.t. a specific model parameter (c,vx or vy)
    dh (float): grid step
    lambda_min (float): standard deviation of the Gaussian filter with which 'grad' will be smoothed
    pad_width (int): number of grid points to pad the boundaries of 'grad' for the FFT
    '''

    grad_padded = np.pad(grad, pad_width = pad_width, mode = 'constant', constant_values = 0)
    
    # Compute Fourier Transform
    fft_grad = np.fft.fft2(grad_padded)

    # Shift the zero frequency component to the center
    fft_grad_shifted = np.fft.fftshift(fft_grad)
    
    # Frequency values
    Nx = grad_padded.shape[0]
    Ny = grad_padded.shape[1]
    kx_vals = np.fft.fftshift(np.fft.fftfreq(Nx, dh)) * 2 * np.pi
    ky_vals = np.fft.fftshift(np.fft.fftfreq(Ny, dh)) * 2 * np.pi
    
    k_max = 1/lambda_min # as in 2D FT of Gaussian Kernel
    
    filt = Gaussfilter(ky_vals, kx_vals, k_max)

    filtered_fft_grad = fft_grad_shifted*filt

    # Reconstruct the array using inverse Fourier transform
    reconstructed_grad_padded = np.fft.ifft2(np.fft.ifftshift(filtered_fft_grad)).real

    reconstructed_grad = reconstructed_grad_padded[pad_width:-pad_width, pad_width:-pad_width]
    
    return reconstructed_grad

def sens_cutmask(nx, ny, ijrec, sigma):
    '''
    Creates a mask to cut out a Gaussian around the receivers.
    Inputs:
    ijrec (ndarray(int,int)): integers representing the position of the receivers on the grid
    nx (int): Number of gridpoints in x-direction
    ny (int): Number of gridpoints in y-dircetion
    sigma (float): Std of cutting Gaussian
    '''
    mask = np.ones((nx,ny))

    xInd, yInd = np.indices([nx,ny])

    irec = ijrec[0]
    jrec = ijrec[1]

    for i,j in zip(irec, jrec):
        dist_sq = (xInd-i)**2 + (yInd-j)**2

        DoI = np.where(np.sqrt(dist_sq) > sigma, 1, 0) # Domain of Interest - Outside of Circles around sources
        mask *= DoI
        
    return mask
        
#############################################################################

def computegrad2D(inpar, ijsrc, mod, adjsrc, p_final, saveh5=True,
                     outfileh5="acoustic_snapshots_reverse.h5"):
    """
    Solve the time reversed acoustic wave equation and adjoint equation in 2D using finite differences on a staggered grid.
    Then integrate the product of the solutions for all time steps and obtain gradients/sensitivity kernels.
    Wrapper function that saves the gradients and snapshots of the build-up in 'outfileh5'.

    Args:
        inpar (dict): dictionary containing various input parameters

                      * inpar["ntimesteps"] (int) number of time steps
                      * inpar["nx"] (int) number of grid nodes in the x direction
                      * inpar["ny"] (int) number of grid nodes in the y direction
                      * inpar["dt"] (float) time step for the simulation
                      * inpar["dh"] (float) grid spacing (same in x and y)
                      * inpar["T0"] (float) typical dim. background temperature
                      * inpar["DeltaT0"] (float) typical dim. temperature anomaly
                      * inpar["Ma"] (float) Mach Number
                      * inpar["savesnapshot"] (bool) switch to save snapshots of the entire wavefield
                      * inpar["savefinal"] (bool) switch to save final two snapshots of the entire wavefield
                      * inpar["snapevery"] (int) save snapshots every "snapevery" iterations
                      * inpar["comp_fact"] (int) compression factor for saved images
                      * inpar["boundcond"] (string) Type of boundary conditions - only "ReflBou" available for now
                      * inpar["constrel"] (string) Type of constituive relationship - only "IdealGas" available for now
        ijsrc (ndarray(int,int)): integers representing the position of the source on the grid
        mod (ndarray(nx,ny)): two-dimensional DeltaT and flow-velocity model (3x(nx)x(ny)-array, Axis 0 has Dim 3 - DeltaT,vx,vy)
        adjsrc (ndarray): source time function
        p_final (ndarray(2,nx,ny)): 2D pressure field at pre-final and final time step (index 0 and 1 on axis 0, resp.) - default None
        saveh5 (bool): whether to save results to HDF5 file or not
        outfileh5 (string): name of the output HDF5 file

    Returns:
        grad (ndarray(3,nx,ny)): gradients - axis 0: indices 0,1,2, -> w.r.t. c, vx, vy, respectively

    """
    ## Selecting according Boundary Condition
    
    # reversing the stf
    adjsrc_rev = adjsrc[:,::-1].copy()
    
    
    if inpar["boundcond"]=="ReflBou" :
        grad, psave, psave_dag, gradsave = _computegrad2D_ReflBound( inpar, ijsrc, mod, p_final, adjsrc_rev)

    else :
        raise("Wrong boundary condition type")

    ##############################
    if saveh5:
        ## save stuff
        hf = h5.File(outfileh5,"w")
        if inpar["savesnapshot"]==True :
            hf["press"] = psave
            hf["press_adj"] = psave_dag
            hf["grad_anim"] = gradsave
            hf["snapevery"] = inpar["snapevery"]
            hf["comp_fact"] = inpar["comp_fact"]
        hf["gradient_c"] = grad[0,:,:]
        hf["gradient_vx"] = grad[1,:,:]
        hf["gradient_vy"] = grad[2,:,:]
        hf["srctf"] = adjsrc_rev
        hf["dh"] = inpar["dh"]
        hf["dt"] = inpar["dt"]
        hf["nx"] = inpar["nx"]
        hf["ny"] = inpar["ny"]
        hf["srcij"] = ijsrc
        hf.close()
        print("Saved acoustic simulation and parameters to ",outfileh5)

    return grad

#########################################################
def _computegrad2D_ReflBound( inpar, ijsrc, mod, p_final, adjsrc_rev) :
    """
    Solve the acoustic wave equation in 2D using finite differences on a staggered grid. 
    Reflective boundary conditions.

    Args:
        inpar (dict): dictionary containing various input parameters (see 'computegrad2D' above)
        ijsrc (ndarray(int,int)): integers representing the position of the source on the grid
        mod (ndarray(nx,ny)): 2D DeltaT and flow-velocity model (3x(nx)x(ny)-array, Axis 0 has dim. 3 - DeltaT,vx,vy)
        p_final (ndarray(2,nx,ny)): 2D pressure field at pre-final and final time step (index 0 and 1 on axis 0, resp.) - default None
        adjsrc_rev (ndarray): adjoint source time function in reverse

    Returns:
        grad (ndarray(3,nx,ny)): gradients - axis 0: indices 0,1,2, -> w.r.t. c, vx, vy, respectively
        
        if inpar["savesnapshot"]==True:
        psave (ndarray): set of snapshots of the wavefield 
        psave_dag (ndarray): snapshots of the adjoint wavefield
        gradsave (ndarray): set of snapshots of the 'unfinished' gradient while forming
        
        
    """
    
    
    ## Loading according constituive relationships
    sys.path.append(
    '../ConstituiveRelationships')
    
    if inpar["constrel"] == 'IdealGas':
        from CR_IdealGas import c2
        
    else:
        raise('wrong constituive relationship')
    
    
    assert(inpar["boundcond"]=="ReflBou")
    print("Starting GRADIENT computation using ADJOINT simulation.")

    ##############################
    # Read Parameters
    
    ## number of grid-nodes in x- & z-direction
    nx = inpar["nx"]
    ny = inpar["ny"]
    
    dh = inpar["dh"]
    dx = dh
    dz = dh
    dt = inpar["dt"]

    T0 = inpar["T0"]
    DeltaT0 = inpar["DeltaT0"]
    Ma = inpar["Ma"]
    ##############################
    
    ## importing the model
    DeltaT = mod[0,:,:]
    vx = mod[1,:,:]
    vy = mod[2,:,:]
    
    ##############################
    #Wavespeed:
    vel = np.sqrt(c2(DeltaT, T0, DeltaT0))
    
    ##############################
    ## Check stability criterion
    ##############################
    maxvp = vel.max()
    
    assert(maxvp*dt*np.sqrt(1/dx**2 + 1/dz**2) < 1.0)
    
    
    ##############################
    ## Arrays to export snapshots
    if inpar["savesnapshot"]==True :
        # size of saved snapshots
        comp_fact = inpar["comp_fact"]
        
        save_nx = nx//comp_fact
        save_ny = ny//comp_fact
        
        ntsave = inpar["ntimesteps"]//inpar["snapevery"]
        psave = np.zeros((save_nx,save_ny,ntsave+1)) # 3D-Array
        tsave=1
        
        ntsave = inpar["ntimesteps"]//inpar["snapevery"]
        psave = np.zeros((save_nx,save_ny,ntsave+1)) # 3D-Array
        psave_dag = np.zeros((save_nx, save_ny,ntsave+1))
        gradsave = np.zeros((3,save_nx,save_ny,ntsave+1))
        tsave=1
    ##############################

    ##############################
    # Reading source time function and source locations
    #nt = adjsrc_rev.size
    isrc = ijsrc[0]
    jsrc = ijsrc[1]
    ##############################
    
    ##############################
    ## Initialize computational arrays & variables
    
    # forward wavefield - to be reconstructed
    pold = np.zeros((nx,ny))
    pcur = p_final[0,:,:] # final condition (nt-1)
    pnew = p_final[1,:,:] # final condition (nt)
    
    # adjont wavefield
    pold_dag = np.zeros((nx,ny))
    pcur_dag = np.zeros((nx,ny))
    pnew_dag = np.zeros((nx,ny))
    
    # gradient
    grad = np.zeros((3,nx,ny))
    
    # create torch tensors
    pcur = torch.tensor(pcur)
    pold = torch.tensor(pold)
    pnew = torch.tensor(pnew)
    pcur_dag = torch.tensor(pcur_dag)
    pold_dag = torch.tensor(pold_dag)
    pnew_dag = torch.tensor(pnew_dag)
    grad = torch.tensor(grad)
    vel = torch.tensor(vel)
    vx = torch.tensor(vx)
    vy = torch.tensor(vy)
    adjsrc_rev = torch.tensor(adjsrc_rev)
    
    if inpar["savesnapshot"]==True:
        psave = torch.tensor(psave)
        psave_dag = torch.tensor(psave_dag)
        gradsave = torch.tensor(gradsave)
    
    # Factors for Solver
    fact_c  = vel[1:-1,1:-1]**2 * (dt**2/dh**2)
    fact_vx = 2*Ma*vx[1:-1,1:-1] * (dt/dh)
    fact_vy = 2*Ma*vy[1:-1,1:-1] * (dt/dh)
    mean_vel_sq = torch.mean(vel)**2
    # Factors for Gradient Integral
    factc_dc = -2/vel[1:-1,1:-1]**3/dt**2
    factvx_dc = -4*Ma*vx[1:-1,1:-1]/vel[1:-1,1:-1]**3/dt/dh
    factvy_dc = -4*Ma*vy[1:-1,1:-1]/vel[1:-1,1:-1]**3/dt/dh
    fact_dv = 2*Ma/vel[1:-1,1:-1]**2/dt/dh
    ################################
    
    # sending all torch tensors to the cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print("Processing on", device)
    
    fact_c = fact_c.to(device)
    fact_vx = fact_vx.to(device)
    fact_vy = fact_vy.to(device)
    factc_dc = factc_dc.to(device)
    factvx_dc = factvx_dc.to(device)
    factvy_dc = factvy_dc.to(device)
    fact_dv = fact_dv.to(device)
    adjsrc_rev = adjsrc_rev.to(device)
    grad = grad.to(device)
    
    pcur = pcur.to(device)
    pold = pold.to(device)
    pnew = pnew.to(device)
    pcur_dag = pcur_dag.to(device)
    pold_dag = pold_dag.to(device)
    pnew_dag = pnew_dag.to(device)
    
    
    ################################
    # Time loop
    for t in range(inpar["ntimesteps"]-1):

        # output of current progress (every 100 time steps)
        if t%100==0 :
            sys.stdout.write("\r Time step {} of {}".format(t,inpar["ntimesteps"]))
            sys.stdout.flush() 
            
        ##=================================================
        # Simulate the adjoint wavefield
        ##=================================================
        ## second order stencil
        dp2dx2_dag = pcur_dag[2:,1:-1]-2.0*pcur_dag[1:-1,1:-1]+pcur_dag[:-2,1:-1]
        dp2dz2_dag = pcur_dag[1:-1,2:]-2.0*pcur_dag[1:-1,1:-1]+pcur_dag[1:-1,:-2]
        ## first order stencil
        dpdx_new_dag = (pnew_dag[2:,1:-1]-pnew_dag[:-2,1:-1])/2
        dpdz_new_dag = (pnew_dag[1:-1,2:]-pnew_dag[1:-1,:-2])/2
        dpdx_dag = (pcur_dag[2:,1:-1]-pcur_dag[:-2,1:-1])/2
        dpdz_dag = (pcur_dag[1:-1,2:]-pcur_dag[1:-1,:-2])/2

        # Inject the Adjoint Source
        s = 0 # source index
        for i, j in zip(isrc, jsrc):
            pcur_dag[i, j] += dt**2 * adjsrc_rev[s, t+1] * mean_vel_sq / dh**2
            s += 1 # ... counting
            
        ## update pressure
        pold_dag[1:-1,1:-1] = (2.0*pcur_dag[1:-1,1:-1] -pnew_dag[1:-1,1:-1]
                               + fact_c*(dp2dx2_dag) 
                               + fact_c*(dp2dz2_dag)
                               - fact_vx*(dpdx_new_dag-dpdx_dag) # Added advection effects
                               - fact_vy*(dpdz_new_dag-dpdz_dag))
        
        ## no-penetration condition (zero normal derivative)
        #vertical walls
        pold_dag[0,1:-1]  = pold_dag[1,1:-1]  # left wall
        pold_dag[-1,1:-1] = pold_dag[-2,1:-1] # right wall
        # horizontal walls
        pold_dag[:,0]  = pold_dag[:,1]        # upper wall
        pold_dag[:,-1] = pold_dag[:,-2]       # lower wall
        
        
        ##=================================================
        # Reconstruct the forward wavefield - in reverse
        ##=================================================
        ## second order stencil
        dp2dx2 = pcur[2:,1:-1]-2.0*pcur[1:-1,1:-1]+pcur[:-2,1:-1]
        dp2dz2 = pcur[1:-1,2:]-2.0*pcur[1:-1,1:-1]+pcur[1:-1,:-2]
        ## first order stencil
        dpdx_new = (pnew[2:,1:-1]-pnew[:-2,1:-1])/2
        dpdz_new = (pnew[1:-1,2:]-pnew[1:-1,:-2])/2
        dpdx = (pcur[2:,1:-1]-pcur[:-2,1:-1])/2
        dpdz = (pcur[1:-1,2:]-pcur[1:-1,:-2])/2
        
        ## update pressure
        pold[1:-1,1:-1] = (2.0*pcur[1:-1,1:-1] -pnew[1:-1,1:-1]
                           + fact_c*(dp2dx2) 
                           + fact_c*(dp2dz2)
                           - fact_vx*(dpdx_new-dpdx) # Added advection effects
                           - fact_vy*(dpdz_new-dpdz))
        
        ## no-penetration condition (zero normal derivative)
        #vertical walls
        pold[0,1:-1]  = pold[1,1:-1]  # left wall
        pold[-1,1:-1] = pold[-2,1:-1] # right wall
        # horizontal walls
        pold[:,0]  = pold[:,1]        # upper wall
        pold[:,-1] = pold[:,-2]       # lower wall
        
        ##=================================================
        # Compute the Gradients
        grad[0,1:-1,1:-1] += pcur_dag[1:-1,1:-1]*(factc_dc*(pnew[1:-1,1:-1]-2*pcur[1:-1,1:-1]+pold[1:-1,1:-1])
                                 +factvx_dc*(dpdx_new-dpdx)
                                 +factvy_dc*(dpdz_new-dpdz))*dt
        
        grad[1,1:-1,1:-1] += pcur_dag[1:-1,1:-1]*fact_dv*(dpdx_new-dpdx)*dt
        
        grad[2,1:-1,1:-1] += pcur_dag[1:-1,1:-1]*fact_dv*(dpdz_new-dpdz)*dt
        
        ##=================================================
        # Storing the data of pnew to be consistent with forward model
        
        #### save snapshots
        if (inpar["savesnapshot"]==True) and (t%inpar["snapevery"]==0):
            psave[:,:,tsave] = pnew.reshape(save_nx, comp_fact, save_ny, comp_fact).mean(dim = (1,3)).cpu()
            psave_dag[:,:,tsave] = pnew_dag.reshape(save_nx, comp_fact, save_ny, comp_fact).mean(dim = (1,3)).cpu()
            gradsave[0,:,:,tsave] = grad[0,:,:].reshape(save_nx, comp_fact, save_ny, comp_fact).mean(dim = (1,3)).cpu()
            gradsave[1,:,:,tsave] = grad[1,:,:].reshape(save_nx, comp_fact, save_ny, comp_fact).mean(dim = (1,3)).cpu()
            gradsave[2,:,:,tsave] = grad[2,:,:].reshape(save_nx, comp_fact, save_ny, comp_fact).mean(dim = (1,3)).cpu()
            tsave += 1
        #==================================================
        
        ## assign the new pold and pcur
        pnew[:,:] = pcur[:,:]
        pcur[:,:] = pold[:,:]
        # as well as adjoint
        pnew_dag[:,:] = pcur_dag[:,:]
        pcur_dag[:,:] = pold_dag[:,:]
        
    
    ##=================================================
    # Storing the last time step
    if (inpar["savesnapshot"]==True) and ((t+1)%inpar["snapevery"]==0):
        psave[:,:,tsave] = pnew.reshape(save_nx, comp_fact, save_ny, comp_fact).mean(dim = (1,3)).cpu()
        psave_dag[:,:,tsave] = pnew_dag.reshape(save_nx, comp_fact, save_ny, comp_fact).mean(dim = (1,3)).cpu()
        gradsave[0,:,:,tsave] = grad[0,:,:].reshape(save_nx, comp_fact, save_ny, comp_fact).mean(dim = (1,3)).cpu()
        gradsave[1,:,:,tsave] = grad[1,:,:].reshape(save_nx, comp_fact, save_ny, comp_fact).mean(dim = (1,3)).cpu()
        gradsave[2,:,:,tsave] = grad[2,:,:].reshape(save_nx, comp_fact, save_ny, comp_fact).mean(dim = (1,3)).cpu()
    ##================================
    print(" ")
    
    if inpar["savesnapshot"]==True:
        psave = psave.numpy()
        psave_dag = psave_dag.numpy()
        gradsave = gradsave.numpy()
    else:
        psave = None
        psave_dag = None
        gradsave = None
        
    # convert gradient to numpy
    grad = grad.cpu()
    grad = grad.numpy()
    
        
        
    return grad, psave, psave_dag, gradsave

#########################################################