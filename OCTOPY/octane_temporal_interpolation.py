#Name: octane_temporal_interpolation.py
#Purpose: A function to determine the warped flows for temporal interpolation
#   
#Inputs:
# im1/2: two images to interpolate (should be brightness temperatures)
# u/v: optical flow motions from initial image pair to interpolate to
# time: normalized (0-1) time value (or array) to warp u and v to
#Returns:
# warped ut/vt arrays for temporal interpolation 
from numba import jit
import numpy as np 
import time, sys

@jit(nopython=True)
def oct_bc(x,nx,bc):
    bc = False
    result = x
    if(x < 0):
        result = 0
        bc = True
    if(x >= nx):
        result = nx-1
        bc = True
    return result

@jit(nopython=True)
def oct_warpflow(u,v,ut,vt,sos, holecount, im1, im2,ttime,nx,ny, donowcast=False):
    #returns warped flow fields
    bc = False
    for l in range(0,2):
        for k in range(0,2):
            for i in range(0,nx-1):
                for j in range(0,ny-1):
                    iv = oct_bc(round(i+ttime[i,j]*v[i,j]),nx-1,bc)
                    jv = oct_bc(round(j+ttime[i,j]*u[i,j]),ny-1,bc)
                    ivl = iv+l
                    jvk = jv+k
                    
                    if(donowcast):
                        imgdiff2 = im1[i,j]
                    else:
                        iv2 = oct_bc(round(i+v[i,j]),nx-1,bc)
                        jv2 = oct_bc(round(j+u[i,j]),ny-1,bc)
                        imgdiff2 = (im1[i,j] - im2[iv2+l,jv2+k])**2
                    if(sos[ivl,jvk] < 0):
                        ut[ivl,jvk] = u[i,j]
                        vt[ivl,jvk] = v[i,j]
                        sos[ivl,jvk] = imgdiff2
                        holecount = holecount - 1
                    else:
                        if((sos[ivl,jvk] > imgdiff2)):
                            ut[ivl,jvk] = u[i,j]
                            vt[ivl,jvk] = v[i,j]
                            sos[ivl,jvk] = imgdiff2

    return ut, vt, sos, holecount
    
    
@jit(nopython=True)
def octane_temporal_interpolation(im1,im2, u,v,time,donowcast=False):
    ut = np.copy(u)
    vt = np.copy(v)
    sos = np.copy(u)
    ut2 = np.copy(u)
    vt2 = np.copy(v)
    sos2 = np.copy(u)
    ut[:,:] = 0. 
    vt[:,:] = 0. 
    sos[:,:] = -1.
    ut2[:,:] = 0.
    vt2[:,:] = 0.
    sos2[:,:] = -1.
    nx, ny = u.shape
    holecount = nx * ny
    holecount2 = nx * ny
    ut, vt, sos,holecount = oct_warpflow(u,v,ut,vt,sos, holecount, im1, im2,time,nx,ny,donowcast=donowcast)
    
    #fill the holes with alternating direction average filter
    rev = True
    while(holecount > 0):
        for jv in range(0,ny):
            if(rev):
                j = jv
            else:
                j = ny - 1 - jv
            for iv in range(0,nx):
                if(rev):
                    i = iv
                else:
                    i = nx - 1 - iv
                if(sos[i,j] < 0):
                    #this is a hole, fill with average of surrounding defined points
                    bc = False
                    imin = oct_bc(i-1,nx-1,bc)
                    imax = oct_bc(i+2,nx-1,bc)
                    jmin = oct_bc(j-1,ny-1,bc)
                    jmax = oct_bc(j+2,ny-1,bc)
                    utsum = 0
                    vtsum = 0
                    utnum = 0
                    for il in range(imin,imax+1):
                        for jl in range(jmin,jmax+1):
                            if(sos[il,jl] >= 0):
                                utsum = utsum+ut[il,jl]
                                vtsum = vtsum+vt[il,jl]
                                utnum = utnum + 1
                    if(utnum > 0):
                        ut[i,j] = utsum/utnum
                        vt[i,j] = vtsum/utnum 
                        sos[i,j] = 0. 
                        holecount = holecount -1 
        if(rev):
            rev = False
        else:
            rev = True
    return ut, vt
    
    
    
    
    
    
    
