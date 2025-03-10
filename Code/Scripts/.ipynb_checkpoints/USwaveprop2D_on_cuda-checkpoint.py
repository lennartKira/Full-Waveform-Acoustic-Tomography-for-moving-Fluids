
#------------------------------------------------------------------------
#
#    PestoSeis, a numerical laboratory to learn about seismology, written
#    in the Python language.
#    Copyright (C) 2021  Andrea Zunino 
#
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#------------------------------------------------------------------------

"""
Functions to calculate acoustic wave propagation in 2D
"""

#######################################################################
#######################################################################

import numpy as np
import sys
import h5py as h5
import torch


#############################################################################
#############################################################################

def _bilinear_interp(f,hgrid, pt):
    """
     Bilinear interpolation (2D).
    """
    xreq=pt[0]
    yreq=pt[1]
    xh=xreq/hgrid
    yh=yreq/hgrid
    i=int(np.floor(xh)) # index starts from 0 so no +1
    j=int(np.floor(yh)) # index starts from 0 so no +1
    xd=xh-i # index starts from 0 so no +1
    yd=yh-j # index starts from 0 so no +1
    intval=f[i,j]*(1.0-xd)*(1.0-yd)+f[i+1,j]*(1.0-yd)*xd+f[i,j+1]*(1.0-xd)*yd+f[i+1,j+1]*xd*yd
    #print xreq,yreq,xh,yh,i,j,xd,yd    
    return intval

# source distribution on grid points
def src_dist(i, sigma):
    f = np.exp(-(i/sigma)**2.)
    return f

#############################################################################

def solveacoustic2D( inpar, ijsrc, mod, sourcetf, srcdom, sourcedf, ijrec, saveh5=True,
                     outfileh5="acoustic_snapshots.h5"):
    """
    Solve the acoustic wave equation in 2D using finite differences on a staggered grid. 
    Wrapper function for various boundary conditions.

    Args:
        inpar (dict): dictionary containing various input parameters

                      * inpar["ntimesteps"] (int) number of time steps
                      * inpar["nx"] (int) number of grid nodes in the x direction
                      * inpar["ny"] (int) number of grid nodes in the z direction
                      * inpar["dt"] (float) time step for the simulation
                      * inpar["dh"] (float) grid spacing (same in x and z)
                      * inpar["T0"] (float) typical dim. background temperature
                      * inpar["DeltaT0"] (float) typical dim. temperature anomaly
                      * inpar["A"] (float) expansion - activate optionally
                      * inpar["Ma"] (float) Mach Number
                      * inpar["savesnapshot"] (bool) switch to save snapshots of the entire wavefield
                      * inpar["savefinal"] (bool) switch to save final two snapshots of the entire wavefield
                      * inpar["snapevery"] (int) save snapshots every "snapevery" iterations
                      * inpar["v_inc"] (int) only plot every v_inc arrow ot the velocity field in simulation
                      * inpar["freesurface"] (bool) True for free surface boundary condition at the top, False for PML
                      * inpar["boundcond"] (string) Type of boundary conditions "PML","GaussTap" or "ReflBou"
                      * inpar["constrel"] (string) Type of constituive relationship "IdealGas", "Water"
        ijsrc (ndarray(2,N_sensors)): integers representing the position of the source on the grid
        mod (ndarray(nx,ny)): two-dimensional DeltaT and flow-velocity model (3x(nx)x(ny)-array, Axis 0 has Dim 3 - DeltaT,vx,vy)
        sourcetf (ndarray(N_sensors, ntimesteps)): source time function for each sensor
        srcdom (ndarray(N_sensors, int): indices of source domain around sources
        sourcedf (ndarray(N_sensors, int): Array (float) determining magnitude of source at indeces determined by srcdom
        ijrec (ndarray(2,N_sensors)): integers representing the position of the receivers on the grid
        saveh5 (bool): whether to save results to HDF5 file or not
        outfileh5 (string): name of the output HDF5 file

    Returns:
        seism (ndarray): seismograms recorded at the receivers
        psave (ndarray): set of snapshots of the wavefield (if inpar["savesnapshot"]==True)

    """
    
    
    ## Selecting according Boundary Condition
    
    if inpar["boundcond"]=="ReflBou" :
        seism,psave,p_final = _solveacouwaveq2D_ReflBound( inpar, ijsrc, mod, sourcetf, srcdom, sourcedf, ijrec )
    
    #elif inpar["boundcond"]=="PML" :
        #seism,psave,p_final = _solveacouwaveq2D_CPML( inpar, ijsrc, mod, sourcetf, srcdom, sourcedf, ijrec ) 

    #elif inpar["boundcond"]=="GaussTap" :
        #seism,psave,p_final = _solveacouwaveq2D_GaussTaper( inpar, ijsrc, mod, sourcetf, srcdom, sourcedf, ijrec ) 

    else :
        raise("Wrong boundary condition type")

    ##############################
    if saveh5:
        ## save stuff
        hf = h5.File(outfileh5,"w")
        if inpar["savesnapshot"]==True :
            hf["press"] = psave
            hf["snapevery"] = inpar["snapevery"]
        if inpar["savefinal"] == True:
            hf["p_final"] = p_final
        hf["seism"] = seism
        hf["DeltaT"] = mod[0,:,:] 
        hf["vx"] = mod[1,:,:]
        hf["vy"] = mod[2,:,:]
        hf["sourcetf"] = sourcetf
        hf["sourcedf"] = sourcedf
        hf["srcdom"] = srcdom
        hf["dh"] = inpar["dh"]
        hf["dt"] = inpar["dt"]
        hf["nx"] = inpar["nx"]
        hf["ny"] = inpar["ny"]
        hf["Ma"] = inpar["Ma"]
        hf["DeltaT0"] = inpar["DeltaT0"]
        hf["T0"] = inpar["T0"]
        hf["v_inc"] = inpar["v_inc"]
        hf["recij"] = ijrec
        hf["srcij"] = ijsrc
        hf.close()
        print("Saved acoustic simulation and parameters to ",outfileh5)

    return seism,psave,p_final


#############################################################################

def _solveacouwaveq2D_ReflBound( inpar, ijsrc, mod, sourcetf, srcdom, sourcedf, ijrec ) :
    """
    Solve the acoustic wave equation in 2D using finite differences on a staggered grid. 
    Reflective boundary conditions.

    Args:
        inpar (dict): dictionary containing various input parameters:

                      * inpar["ntimesteps"] (int) number of time steps
                      * inpar["nx"] (int) number of grid nodes in the x direction
                      * inpar["ny"] (int) number of grid nodes in the z direction
                      * inpar["dt"] (float) time step for the simulation
                      * inpar["dh"] (float) grid spacing (same in x and z)
                      * inpar["T0"] (float) typical dim. background temperature
                      * inpar["DeltaT0"] (float) typical dim. temperature anomaly
                      * inpar["A"] (float) expansion - activate optionally
                      * inpar["Ma"] (float) Mach Number
                      * inpar["savesnapshot"] (bool) switch to save snapshots of the entire wavefield
                      * inpar["savefinal"] (bool) switch to save final two snapshots of the entire wavefield
                      * inpar["snapevery"] (int) save snapshots every "snapevery" iterations
                      * inpar["v_inc"] (int) only plot every v_inc arrow ot the velocity field in simulation
                      * inpar["freesurface"] (bool) True for free surface boundary condition at the top, False for PML
                      * inpar["boundcond"] (string) Type of boundary conditions "PML","GaussTap" or "ReflBou"
                      * inpar["constrel"] (string) Type of constituive relationship "IdealGas", "Water"
        ijsrc (ndarray(2,N_sensors)): integers representing the position of the source on the grid
        mod (ndarray(nx,ny)): two-dimensional DeltaT and flow-velocity model (3x(nx)x(ny)-array, Axis 0 has Dim 3 - DeltaT,vx,vy)
        sourcetf (ndarray(N_sensors, ntimesteps)): source time function for each sensor
        srcdom (ndarray(N_sensors, int): indices of source domain around sources
        sourcedf (ndarray(N_sensors, int): Array (float) determining magnitude of source at indeces determined by srcdom
        ijrec (ndarray(2,N_sensors)): integers representing the position of the receivers on the grid

    Returns:
        seism (ndarray): seismograms recorded at the receivers
        psave (ndarray): set of snapshots of the wavefield (if inpar["savesnapshot"]==True)

    """
    
    
    ## Loading according constituive relationships - here also the expansion coefficients might be loaded!
    sys.path.append(
    '../ConstituiveRelationships')
    
    if inpar["constrel"] == 'IdealGas':
        from CR_IdealGas import c2
        
    elif inpar["constrel"] == 'Water':
        from CR_Water import c2
        
    else:
        raise('wrong constituive relationship')
    
    
    assert(inpar["boundcond"]=="ReflBou")
    print("Starting ACOUSTIC solver with reflective boundaries all around.")

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
    #A = inpar["A"]
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
    print(" Stability criterion, CFL number:",(maxvp*dt*np.sqrt(1/dx**2+1/dz**2))) # WHY IS THAT NEEDED @Christian
    assert(maxvp*dt*np.sqrt(1/dx**2 + 1/dz**2) < 1.0)
    
    
    ##############################
    ## Arrays to export snapshots
    if inpar["savesnapshot"]==True :
        # size of saved snapshots
        save_nx = 512
        save_ny = 512
        
        ratio_nx = nx//save_nx
        ratio_ny = ny//save_ny
        
        ntsave = inpar["ntimesteps"]//inpar["snapevery"]
        psave = np.zeros((save_nx,save_ny,ntsave+1)) # 3D-Array
        tsave=1
        
    if inpar["savefinal"]==True :
        p_final = np.zeros((2, inpar["nx"],inpar["ny"])) # 3D-Array
        tfin=0

    ## Arrays to return seismograms
    nrecs = ijrec.shape[1]
    receiv = np.zeros((nrecs,inpar["ntimesteps"]))
    ##############################

    ##############################
    # Reading source time function and source location
    #nt = sourcetf.size
    isrc = ijsrc[0]
    jsrc = ijsrc[1]
    ##############################
    
    ##############################
    ## Initialize computational arrays & variables
    
    pcur = np.zeros((nx,ny))
    pold = np.zeros((nx,ny))
    pnew = np.zeros((nx,ny))
    
    # create torch tensors
    pcur = torch.tensor(pcur)
    pold = torch.tensor(pold)
    pnew = torch.tensor(pnew)
    vel = torch.tensor(vel)
    vx = torch.tensor(vx)
    vy = torch.tensor(vy)
    receiv = torch.tensor(receiv)
    sourcetf = torch.tensor(sourcetf)
    sourcedf = torch.tensor(sourcedf)
    

    fact_c  = vel[1:-1,1:-1]**2 * (dt**2/dh**2)
    fact_vx = 2*Ma*vx[1:-1,1:-1] * (dt/dh)
    fact_vy = 2*Ma*vy[1:-1,1:-1] * (dt/dh)
    mean_vel_sq = torch.mean(vel)**2
    ################################
    
    # throwing all torch tensors on the cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Processing on', device)
    
    fact_c = fact_c.to(device)
    fact_vx = fact_vx.to(device)
    fact_vy = fact_vy.to(device)
    sourcetf = sourcetf.to(device)
    sourcedf = sourcedf.to(device)
    
    pcur = pcur.to(device)
    pold = pold.to(device)
    pnew = pnew.to(device)
    
    receiv = receiv.to(device)
    
    if inpar["savesnapshot"]==True:
        psave = torch.tensor(psave)
        #psave = psave.to(device)
        
    if inpar["savefinal"]==True:
        p_final = torch.tensor(p_final)
        #p_final = p_final.to(device)
    
    
    # Is advection effect activated?
    #print('ADVECTION EFFECT IS DEACTIVATED!')
    ################################
    # Time loop
    #print((" Time step dt: {}".format(dt)))

    for t in range(inpar["ntimesteps"]):

        # output of current progress (every 100 time steps)
        if t%100==0 :
            sys.stdout.write("\r Time step {} of {}".format(t,inpar["ntimesteps"]))
            sys.stdout.flush()

        ##=================================================
        ## second order stencil
        dp2dx2 = pcur[2:,1:-1]-2.0*pcur[1:-1,1:-1]+pcur[:-2,1:-1]
        dp2dz2 = pcur[1:-1,2:]-2.0*pcur[1:-1,1:-1]+pcur[1:-1,:-2]
        ## first order stencil
        dpdx = (pcur[2:,1:-1]-pcur[:-2,1:-1])/2
        dpdz = (pcur[1:-1,2:]-pcur[1:-1,:-2])/2
        dpdx_old = (pold[2:,1:-1]-pold[:-2,1:-1])/2
        dpdz_old = (pold[1:-1,2:]-pold[1:-1,:-2])/2
        #dDTdx = (DeltaT[2:,1:-1]-DeltaT[:-2,1:-1])/2
        #dDTdz = (DeltaT[1:-1,2:]-DeltaT[1:-1,:-2])/2
        
        ## update pressure
        pnew[1:-1,1:-1] = (2.0*pcur[1:-1,1:-1] -pold[1:-1,1:-1]
                           + fact_c*(dp2dx2) 
                           + fact_c*(dp2dz2)
                           - fact_vx*(dpdx-dpdx_old) # Added advection affects
                           - fact_vy*(dpdz-dpdz_old))
                           #- fact_c*A/(1+A*DeltaT[1:-1,1:-1])*(dDTdx*dpdx)  # Added density effects
                           #- fact_c*A/(1+A*DeltaT[1:-1,1:-1])*(dDTdz*dpdz))
        
        ## no-penetration condition (zero normal derivative)
        #vertical walls
        pnew[0,1:-1]  = pnew[1,1:-1]  # left wall
        pnew[-1,1:-1] = pnew[-2,1:-1] # right wall
        # horizontal walls
        pnew[:,0]  = pnew[:,1]        # upper wall
        pnew[:,-1] = pnew[:,-2]       # lower wall
        
        #s = 0 # source index...
        #for i, j in zip(isrc, jsrc):
        #    pnew[i, j] += dt**2 * sourcetf[s, t] * mean_vel_sq / dh**2
        #    s += 1 # ... counting
        
        ## Spacially extended sources
        ################################################################
        # Pre-compute offsets
        offsets_i = srcdom[:, 0, :][:, np.newaxis, :]  # Shape (3, 1, 3)
        offsets_j = srcdom[:, 1, :][:, np.newaxis, :]  # Shape (3, 1, 3)
        
        # Pre-compute the term that is independent of i, j
        update_term = sourcedf * dt**2 * mean_vel_sq / dh**2
        
        # Expand isrc and jsrc to match offsets dimensions
        expanded_isrc = isrc[:, np.newaxis, np.newaxis]  # Shape (3, 1, 1)
        expanded_jsrc = jsrc[:, np.newaxis, np.newaxis]  # Shape (3, 1, 1)
        
        # Calculate all indices
        indices_i = expanded_isrc + offsets_i  # Shape (3, 1, 3)
        indices_j = expanded_jsrc + offsets_j  # Shape (3, 1, 3)
        
        # Flatten indices to use advanced indexing
        indices_i = indices_i.flatten()  # Shape (9,)
        indices_j = indices_j.flatten()  # Shape (9,)
        
        # Flatten sourcetf to match the flattened indices
        update_values = (update_term * sourcetf[:, t][:, np.newaxis]).flatten()  # Shape (9,)
        
        # Perform the update in a single operation
        pnew[indices_i, indices_j] += update_values
        ################################################################
        
        ## assign the new pold and pcur
        pold[:,:] = pcur[:,:]
        pcur[:,:] = pnew[:,:]

        ##=================================================
        
        ##### receivers
        for r in range(nrecs) :
            #rec_press = _bilinear_interp(pcur,dh,recpos[r,:]) # interpolation of receivers aren't on gridpt...
            receiv[r,t] = pcur[ijrec[0,r],ijrec[1,r]]
        
        #### save snapshots
        if (inpar["savesnapshot"]==True) and (t%inpar["snapevery"]==0) :
            psave[:,:,tsave] = pcur.reshape(save_nx, ratio_nx, save_ny, ratio_ny).mean(dim = (1,3)).cpu()
            tsave += 1
            
        ### save final two snapshots
        if (inpar["savefinal"]==True) and (t>=int(inpar["ntimesteps"])-2):
            p_final[tfin,:,:] = pcur.cpu()
            tfin += 1

    ##================================
    print(" ")
    
    #if inpar["savesnapshot"]==False :
    #    psave = None
    
    if inpar["savesnapshot"]==True:
        #psave = psave.cpu()
        psave = psave.numpy()
    else:
        psave = None
        
    if inpar["savefinal"]==True:
        #p_final = p_final.cpu()
        p_final = p_final.numpy()
    else:
        p_final = None
        
    receiv = receiv.cpu()
    receiv = receiv.numpy()
        
    return receiv,psave,p_final

#########################################################