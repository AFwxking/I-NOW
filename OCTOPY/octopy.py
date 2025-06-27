# %%
# //Function: oct_variational_optical_flow.cu
# //Purpose: This is CUDA-based code designed to perform variational optical flow computation given two image objects
# //
# //Author: Jason Apke
#Note, this is a port from cuda to numba/python, hence the weird looking comments
#start with the relavent imports
#import satpy
import numpy as np
from math import *
from numba import cuda
import time
#### GPU kernels ###
# This is a boundary condition function
@cuda.jit(device=True)
def oct_bc_cu(x, nx):
    bc=False
    if(x < 0):
        x = 0
        bc=True
    if(x >= nx):
        x = nx-1
        bc=True
    return x, bc


# //A square function 
@cuda.jit(device=True)
def jsq(x):
    return x*x

# A function to determine the zoom size based on an input factor
@cuda.jit(device=True)
def zoom_size(nx, ny, factor):
    nxx = int((nx* factor + 0.5))
    nyy = int((ny* factor + 0.5))
    return nxx, nyy

#This is a bilinear interpolation function that also outputs coefficients to reduce divisions 
@cuda.jit(device=True)
def oct_binterp_coefs_cu (x, y,x1, x2, y1, y2,f11, f21, f12, f22):
    # //All about efficiency, only compute bilinear terms once
    p1 = (x2-x)/(x2-x1)
    p2 = (x-x1)/(x2-x1)
    p3 = ((y2-y)/(y2-y1))
    p4 = ((y-y1)/(y2-y1))
    return p3*((p1)*f11+(p2)*f21)+p4*((p1)*f12+(p2)*f22),p1,p2,p3,p4

#bilinear interpolation function using precomputed coefficients
@cuda.jit(device=True)
def oct_coef_binterp_cu(p1, p2, p3, p4, f11,f21,f12,f22):
    # //uses the bilinear terms computed from the function above
    return p3*((p1)*f11+(p2)*f21)+p4*((p1)*f12+(p2)*f22)


# //Robust function derivative for smoothness constraint
@cuda.jit(device=True)
def oct_PSI_smooth_cu(x,doq):
    # //doq means do quadratic (for graduated non-convexity minimization)
    if(doq == 0):
        answer= 1./(sqrt((x+1E-6)))
    else:
        answer=1.
    return answer

# //experimental weighting function
@cuda.jit(device=True)
def scw(a, b, sigma):
    # //an experimental smoothness constraint weighting
    amb = a-b
    return exp(-1.*(amb*amb)/sigma)

# //Robust function derivative for data constraint
@cuda.jit(device=True)
def oct_PSI_data_cu(x, doq):
    if(doq == 0):

        answer=1./(sqrt(x+1E-6))
    else:
        answer=1.
    return answer

# //A function for matrix multiplication
@cuda.jit(device=True)
def multiply_row( rowsize,rowstart, Aval, Acol, x0, nx, An):
    sum = 0
    for i in range(rowstart, rowstart+rowsize):
        sum += Aval[i]*x0[Acol[i]]
    return sum

# //A function for multiplying matrices by vectors
@cuda.jit(device=True)
def jMatXVec(  Aval,Arow, ArowSP, Acol, x0, An,nx,ans):
    x = cuda.grid(1) #This should be grid.thread_rank() from cxx
    gs = cuda.gridsize(1) #equivalent to grid_size()
    #(int k = grid.thread_rank(); k < nx; k+=grid.size()) was the for loop syntax for cuda cxx
    for k in range(x,nx,gs):
        row_begin = ArowSP[k]
        if(k < nx-1):
            row_end = ArowSP[k+1]
        else:
            row_end = An
        ans[k] = multiply_row(row_end-row_begin,row_begin,Aval,Acol,x0,nx,An)
# //A quick inverter kernel for the M matrix
@cuda.jit(device=True)
def jDiagInv(M, nx):
    x = cuda.grid(1)
    gs = cuda.gridsize(1)
    for k in range(x,nx,gs):
        M[k] = 1./M[k]

#Note result c is a float array with size of 1!!! (to work with atomic add)
@cuda.jit(device=True)
def jVecXVec(vecA, vecB,resultc,size):
   x = cuda.grid(1) #This should be grid.thread_rank()
   gs = cuda.gridsize(1) #equivalent to grid_size(), I think
   sum = 0.
   for i in range(x,size,gs):
      sum += vecA[i]*vecB[i]
   cuda.atomic.add(resultc,0,sum)

# //A scalar times a vector function NOTE you will need to adjust this call where relavent
@cuda.jit(device=True)
def jScaleXVec(a, vec1,vec2, nx):
    x = cuda.grid(1)
    gs = cuda.gridsize(1)
    for k in range(x,nx,gs):
        vec2[k] = a*vec1[k]  
    return vec2
# //A vector addition function with a bonus d*a scalar multiplication (speeds things up here), vec a, vec b, result c, scalar d, size nx
@cuda.jit(device=True)
def jVecPVec(a, b, c, d, nx):
    x = cuda.grid(1)
    gs = cuda.gridsize(1)
    for k in range(x,nx,gs):
        c[k] = d*a[k]+b[k];  

@cuda.jit(device=True)
def fill_GK(GK, factor,filtsize, filtsize2):
    i = cuda.grid(1)
    if(i == 0):
        #this sigma is currently not a tunable, maybe should be in the future...
        sigma = 0.6*sqrt(1.0/(factor*factor)-1.0) 
        s = 2.0 * sigma * sigma

        sum = 0.0
        for x in range(-filtsize,filtsize+1):
            GK[x + int(filtsize)] = (exp(-(float(x) * x) / s)) / (3.14159265358979323846 * s)
            sum += GK[x + int(filtsize)]

        for j in range(0,filtsize2):
            GK[j] /= sum
    return GK

# //bicubic interpolation sub-function 3
@cuda.jit(device=True)
def oct_cell_cu( v0,v1,v2,v3, x):
    return  v1 + 0.5 * x * (v2 - v0 + x * (2.0 *  v0 - 5.0 * v1 + 4.0 * v2 - v3 + x * (3.0 * (v1 - v2) + v3 - v0)))

# //bicubic interpolation sub-function 2
@cuda.jit(device=True)
def oct_bicubic_cell_cu (p00,p01,p02,p03,p10,p11,p12,p13,p20,p21,p22,p23,p30,p31,p32,p33, x, y):
    v0 = oct_cell_cu(p00,p10,p20,p30, y)
    v1 = oct_cell_cu(p01,p11,p21,p31, y)
    v2 = oct_cell_cu(p02,p12,p22,p32, y)
    v3 = oct_cell_cu(p03,p13,p23,p33, y)
    return oct_cell_cu(v0,v1,v2,v3, x)
# //bicubic interpolation sub-function 1
@cuda.jit(device=True)
def oct_bicubic_cu(input, uu, vv, nx, ny):
    sx = 1
    sy = 1
    x,bc   = oct_bc_cu(float((int(uu))),nx)
    y,bc   = oct_bc_cu(float((int(vv))),ny)
    mx,bc  = oct_bc_cu(float((int((uu-sx)))),nx)
    my,bc  = oct_bc_cu(float((int((vv-sy)))),ny)
    dx,bc  = oct_bc_cu(float((int((uu+sx)))),nx)
    dy,bc  = oct_bc_cu(float((int((vv+sy)))),ny)
    ddx,bc = oct_bc_cu(float((int((uu+2*sx)))),nx)
    ddy,bc = oct_bc_cu(float((int((vv+2*sy)))),ny)
    #convert to ints (needed for python)
    x = int(x)
    y = int(y)
    mx = int(mx)
    my = int(my)
    dx = int(dx)
    dy = int(dy)
    ddx = int(ddx)
    ddy = int(ddy)
    nxtmy = nx*my
    nxty = nx*y
    nxtdy = nx*dy
    nxtddy = nx*ddy
    # //Below may be less than ideal for GPU memory access issues, may cause slowdowns
    # unfortunately NUMBA does not yet support texture memory, which would be faster for whats below
    p11 = input[mx  + nxtmy]
    p12 = input[x   + nxtmy]
    p13 = input[dx  + nxtmy]
    p14 = input[ddx + nxtmy]

    p21 = input[mx  + nxty] #these two have relatively large differences from the c version
    p22 = input[x   + nxty]
    p23 = input[dx  + nxty]
    p24 = input[ddx + nxty] #this and p21 have relatively large differences

    p31 = input[mx  + nxtdy]
    p32 = input[x   + nxtdy]
    p33 = input[dx  + nxtdy]
    p34 = input[ddx + nxtdy]

    p41 = input[mx  + nxtddy]
    p42 = input[x   + nxtddy]
    p43 = input[dx  + nxtddy]
    p44 = input[ddx + nxtddy]

    return oct_bicubic_cell_cu(p11,p12,p13,p14,p21,p22,p23,p24,p31,p32,p33,p34,p41,p42,p43,p44,uu-x,vv-y)



# //A horizontal gaussian convolution smoothing filter
@cuda.jit(device=True)
def convh(image,imageout,GK, nx, ny,nc, factor,filtsize):
    xityi = nx*ny
    bc = False
    x = cuda.grid(1)
    gs = cuda.gridsize(1)
    for lxyza in range(x,xityi*nc,gs):
        c = int(int(lxyza)/int(xityi))
        tv = int(lxyza-c*xityi)
        i = int((tv) % nx)
        wsum = 0.
        for kk2 in range(-filtsize,filtsize):
            iiv,bc = oct_bc_cu(float(i+kk2),nx)
            iiv = int(round(iiv))
            wsum = wsum+GK[kk2+filtsize]*image[lxyza+(iiv-i)]
        imageout[lxyza] = wsum

# //A vertical gaussian convolution smoothing filter
@cuda.jit(device=True)
def convv(imageout,Is,GK, nx, ny,nc, factor,filtsize):
    xityi = nx*ny
    bc = False
    # //Vertical convolution
    x = cuda.grid(1)
    gs = cuda.gridsize(1)
    for lxyza in range(x,xityi*nc,gs):
        c = int(lxyza/xityi)
        tv = int(lxyza-c*xityi)
        i = int((tv) % nx)
        j = int((tv-i)/nx)
        wsum = 0.
        for kk2 in range(-filtsize,filtsize):
            jjv,bc = oct_bc_cu(float(j+kk2),ny)
            jjv = int(round(jjv))
            wsum = wsum+GK[kk2+int(filtsize)]*imageout[lxyza+nx*(jjv-int(j))]
        Is[lxyza] = wsum


# //A zoom out function, where the input image Is is blurred with the gaussian filters above, then subsampled
@cuda.jit(device=True)
def zoom_out (Is,imageout, nx, ny,nc, factor):

    nxx = int(float(nx* factor) + 0.5)
    nyy = int(float(ny* factor) + 0.5)
    nxxtnyy = int(nxx*nyy)
    x = cuda.grid(1)
    gs = cuda.gridsize(1)
    for lxyza in range(x,nxxtnyy*nc,gs):
        c = int(lxyza/nxxtnyy)
        tv = int(lxyza-c*nxxtnyy)
        ii = int((tv) % nxx)
        jj = int((tv-ii)/nxx)
        i2 = int(ii/factor)
        j2 = int(jj/factor)
        imageout[lxyza] =  oct_bicubic_cu(Is,i2,j2, nx, ny)

# //A function to compute the gradients of input image geo1
@cuda.jit(device=True)
def oct_compgrad_cu (geo1,xi,yi,nc,gradxarr,gradyarr):
    xityi = xi*yi
    xityitnc = xityi*nc
    x = cuda.grid(1)
    gs = cuda.gridsize(1)
    for lxyza in range(x,xityitnc,gs):
        c = int(int(lxyza)/int(xityi))
        tv = lxyza-c*xityi
        i = (tv) % xi
        j = (tv-i)/xi # //this should work
        xitj = xi*j
        xityitc = xityi*c
        bc = False
        jp1, bc = oct_bc_cu(float(j+1),yi)
        jp2, bc = oct_bc_cu(float(j+2),yi)
        jm1, bc = oct_bc_cu(float(j-1),yi)
        jm2, bc = oct_bc_cu(float(j-2),yi)
        jp1 = int(jp1)
        jp2 = int(jp2)
        jm1 = int(jm1)
        jm2 = int(jm2)

        ip1, bc = oct_bc_cu(float(i+1),xi)
        ip2, bc = oct_bc_cu(float(i+2),xi)
        im1, bc = oct_bc_cu(float(i-1),xi)
        im2, bc = oct_bc_cu(float(i-2),xi)
        ip1 = int(ip1)
        ip2 = int(ip2)
        im1 = int(im1)
        im2 = int(im2)

        lxyz_pdx = int((ip1)+xitj+xityitc)
        lxyz_pdx2 = int((ip2)+xitj+xityitc)
        lxyz_mdx = int((im1)+xitj+xityitc)
        lxyz_mdx2 = int((im2)+xitj+xityitc)

        lxyz_pdy = int(i+xi*jp1+xityitc)
        lxyz_pdy2 = int(i+xi*jp2+xityitc)
        lxyz_mdy = int(i+xi*jm1+xityitc)
        lxyz_mdy2 = int(i+xi*jm2+xityitc)

        gradxarr[lxyza] = (-geo1[lxyz_pdx2]+8.*geo1[lxyz_pdx]-8.*geo1[lxyz_mdx]+geo1[lxyz_mdx2])/12.0 #; //(dist);

        gradyarr[lxyza] = (-geo1[lxyz_pdy2]+8.*geo1[lxyz_pdy]-8.*geo1[lxyz_mdy]+geo1[lxyz_mdy2])/12.0 #; //(dist);

# //A function to zoom in flow estimates to the next pyramid levels
# //Careful, this zoom in function is SPECIFICALLY for flow, that is, it will multiply the result by input sf
@cuda.jit(device=True)
def zoom_in(flow,flowout, nx, ny, nxx, nyy, sf):
    factorx = (float(nxx) / nx)
    factory = (float(nyy) / ny)
    x = cuda.grid(1)
    gs = cuda.gridsize(1)
    for lxyza in range(x,nxx*nyy,gs):
        ii = int(lxyza) % int(nxx)
        jj = (int(lxyza)-ii)/int(nxx)
        i2 =  float((ii / factorx)-(0.5-0.5/factorx))
        j2 =  float((jj / factory)-(0.5-0.5/factory))
        flowout[lxyza] = oct_bicubic_cu(flow, i2, j2, nx, ny)/sf

# This is the main call to the variational algorithm which uses all the functions above
# Note, there are lots of inputs, and two primary outputs, uval and vval
@cuda.jit
def octane(Aval,Arow, ArowSP, Acol, x0,
        uval, vval, uvalt, vvalt,uhval,vhval,
        geo1,geo2,geo10,geo20,CTHc,CTH, GK,gradxarrg2,
        gradyarrg2,gradxxarr,gradxyarr,gradyyarr,
        gradxarr,gradyarr,nchan,alpha,lambdadalpha,lambdaco,
        An,nx,nx2, ny2,
        bcu,Mval,Mrow,z0,
        p0, residc,tol,iters,rk, zktrk,z0tr0,
        rkTzk,pkTApk,dummyvec,
        liters,kiters,filtsize,filtsize2, scaleFactor,scsig,dozim):
    xio = 0
    yio = 0 #for some reason python needs these defined before it can compile
    grid = cuda.cg.this_grid()
    bc = False

    for k in range(0,kiters):
        factor = scaleFactor**(kiters-k-1)
        xi, yi = zoom_size(nx2,ny2,factor)
        xi2 = 2*xi
        xityi = xi*yi
        xityitnc = xi*yi*nchan
        lambdac = lambdaco*(0.5**(k))
        grid.sync()
        # //Zoom in the previous guess if needed, otherwise fill in the first guess flow values
        if(k > 0):
            zoom_in(uvalt,uval,xio,yio,xi,yi,scaleFactor)
            grid.sync()

            zoom_in(vvalt,vval,xio,yio,xi,yi,scaleFactor)
            grid.sync()
        if(k == kiters-1):
            gtr = cuda.grid(1)
            gs = cuda.gridsize(1)
            for lxyza in range(gtr,xityitnc,gs):
                geo1[lxyza] = geo10[lxyza]
                geo2[lxyza] = geo20[lxyza]
            grid.sync()
            gtr = cuda.grid(1)
            gs = cuda.gridsize(1)
            for lxyza in range(gtr,xityi,gs):
                uvalt[lxyza] = uhval[lxyza]
                vvalt[lxyza] = vhval[lxyza]
            grid.sync()
        else:
            # //Zoom out the images to the level of the pyramid
            sigma = 1.0/sqrt(2.*factor)
            # //capping the blur filter size for speed
            filtsize = int(2*sigma)
            if(filtsize < 5):
                filtsize = 5
            filtsize2 = 2*filtsize+1
            GK = fill_GK(GK,factor,filtsize,filtsize2)
            # //Zoom out geo10
            grid.sync()
            convh(geo10,geo1, GK, nx2, ny2, nchan, factor, filtsize)
            grid.sync()
            convv(geo1,dummyvec, GK, nx2, ny2, nchan, factor, filtsize)
            grid.sync()
            zoom_out (dummyvec,geo1, nx2, ny2,nchan,factor)
            grid.sync()
            convh(geo20,geo2, GK, nx2, ny2, nchan, factor, filtsize)
            grid.sync()
            convv(geo2,dummyvec, GK, nx2, ny2, nchan, factor, filtsize)
            grid.sync()
            zoom_out (dummyvec,geo2, nx2, ny2,nchan,factor)
            grid.sync()
            
            # //Zoom out flows for U hat/V hat value
            convh(uhval,uvalt, GK, nx2, ny2, 1, factor, filtsize)
            grid.sync()
            convv(uvalt,dummyvec, GK, nx2, ny2, 1, factor, filtsize)
            grid.sync()
            zoom_out (dummyvec,uvalt, nx2, ny2,1,factor)
            grid.sync()
            
            convh(vhval,vvalt, GK, nx2, ny2, 1, factor, filtsize)
            grid.sync()
            convv(vvalt,dummyvec, GK, nx2, ny2, 1, factor, filtsize)
            grid.sync()
            zoom_out (dummyvec,vvalt, nx2, ny2,1,factor)
            grid.sync()
            gtr = cuda.grid(1)
            gs = cuda.gridsize(1)
            for lxyza in range(gtr,xityi,gs):
                uvalt[lxyza] *= factor
                vvalt[lxyza] *= factor #//scaling pixel displacements to current zoom level
            grid.sync()


        if(k == 0):
            gtr = cuda.grid(1)
            gs = cuda.gridsize(1)
            for lxyza in range(gtr,xityi,gs):
                uval[lxyza] = uvalt[lxyza] # //set initial estimate to first guess
                vval[lxyza] = vvalt[lxyza]
            grid.sync()
            
        # //Compute the relavant gradients
        oct_compgrad_cu(geo1,xi,yi,nchan,gradxarr,gradyarr)
        grid.sync()
        oct_compgrad_cu(geo2,xi,yi,nchan,gradxarrg2,gradyarrg2)
        grid.sync()
        oct_compgrad_cu(gradxarrg2,xi,yi,nchan,gradxxarr,gradxyarr)
        grid.sync()
        oct_compgrad_cu(gradyarrg2,xi,yi,nchan,gradxyarr,gradyyarr)
        grid.sync()
        nx = xityi*2
        An = 12*xityi-4*xi-4*yi

        # //Graduated non-convexity iterations (always 3 steps), minimize by setting robust functions to quadratic at first stage
        # then switch to half and half at second stage, then full sqrt and/or robust function at final stage
        for gnc in range(0,3):
            al1 = 1.-0.5*gnc
            # //Inner iterations
            for l in range(0,liters):
                gtr = cuda.grid(1)
                gs = cuda.gridsize(1)
                #This part fills the sparse matrix at each step
                for lxyza in range(gtr,xityi,gs):
                    lxyz = lxyza+lxyza
                    lxyzap1 = lxyza+1
                    lxyzam1 = lxyza-1
                    lxyzp1p0 = lxyzap1
                    lxyzp1p1 = lxyzap1+xi
                    lxyzp1m1 = lxyzap1-xi
                    lxyzp0p1 = lxyza+xi
                    lxyzp0m1 = lxyza-xi
                    lxyzm1p0 = lxyzam1
                    lxyzm1p1 = lxyzam1+xi
                    lxyzm1m1 = lxyzam1-xi
                    ii = lxyza % xi
                    jj = (lxyza-ii)/xi

                    if(ii == 0):
                        lxyzm1p0 = lxyzm1p0+2
                        lxyzm1p1 = lxyzm1p1+2
                        lxyzm1m1 = lxyzm1m1+2
                    if(ii == xi-1):
                        lxyzp1p0 = lxyzp1p0-2
                        lxyzp1p1 = lxyzp1p1-2
                        lxyzp1m1 = lxyzp1m1-2
                    if(jj == 0):
                        lxyzp1m1 = lxyzp1m1+xi+xi
                        lxyzp0m1 = lxyzp0m1+xi+xi
                        lxyzm1m1 = lxyzm1m1+xi+xi
                    if(jj == yi-1):
                        lxyzp1p1 = lxyzp1p1-xi-xi
                        lxyzp0p1 = lxyzp0p1-xi-xi
                        lxyzm1p1 = lxyzm1p1-xi-xi

                    up1p0 = uval[lxyzp1p0] 
                    up0p0 = uval[lxyza] 
                    up1p1 = uval[lxyzp1p1] 
                    up1m1 = uval[lxyzp1m1] 
                    up0p1 = uval[lxyzp0p1] 
                    up0m1 = uval[lxyzp0m1] 
                    um1p1 = uval[lxyzm1p1] 
                    um1p0 = uval[lxyzm1p0] 
                    um1m1 = uval[lxyzm1m1]



                    vp1p0 = vval[lxyzp1p0] 
                    vp0p0 = vval[lxyza] 
                    vp1p1 = vval[lxyzp1p1] 
                    vp1m1 = vval[lxyzp1m1] 
                    vp0p1 = vval[lxyzp0p1] 
                    vp0m1 = vval[lxyzp0m1] 
                    vm1p1 = vval[lxyzm1p1] 
                    vm1p0 = vval[lxyzm1p0] 
                    vm1m1 = vval[lxyzm1m1] 


                    Uip1 = jsq(up1p0-up0p0)+jsq(0.25*((up1p1-up1m1) + (up0p1-up0m1)))+jsq(vp1p0-vp0p0)+jsq(0.25*((vp1p1-vp1m1) + (vp0p1-vp0m1)))
                    Uim1 = jsq(up0p0-um1p0)+jsq(0.25*((um1p1-um1m1) + (up0p1-up0m1)))+jsq(vp0p0-vm1p0)+jsq(0.25*((vm1p1-vm1m1) + (vp0p1-vp0m1)))
                    Ujp1 = jsq(up0p1-up0p0)+jsq(0.25*((up1p1-um1p1) + (up1p0-um1p0)))+jsq(vp0p1-vp0p0)+jsq(0.25*((vp1p1-vm1p1) + (vp1p0-vm1p0)))
                    Ujm1 = jsq(up0p0-up0m1)+jsq(0.25*((up1m1-um1m1) + (up1p0-um1p0)))+jsq(vp0p0-vp0m1)+jsq(0.25*((vp1m1-vm1m1) + (vp1p0-vm1p0)))


                    psis1 = oct_PSI_smooth_cu(Uim1,0)
                    psis2 = oct_PSI_smooth_cu(Ujm1,0)
                    psis3 = oct_PSI_smooth_cu(Uip1,0)
                    psis4 = oct_PSI_smooth_cu(Ujp1,0)
                    psistot = psis1+psis2+psis3+psis4
                    psistotq = 4. 
                    psisnmiu = psis1*(um1p0)+psis2*(up0m1) + psis3*(up1p0)+psis4*(up0p1)
                    psisnmiv = psis1*(vm1p0)+psis2*(vp0m1) + psis3*(vp1p0)+psis4*(vp0p1)

                    psisnmiuq = um1p0 + up0m1 + up1p0 + up0p1 
                    psisnmivq = vm1p0 + vp0m1 + vp1p0 + vp0p1 
                    vr1, vr2, vr3, vr4, vr5, vr6, intcomp        = 0, 0, 0, 0, 0, 0, 0
                    vr12, vr22, vr32, vr42, vr52, vr62, intcomp2 = 0, 0, 0, 0, 0, 0, 0
                    bc =False
                    bc2=False
                    bc3=False
                    iv, bc = oct_bc_cu(float(ii+up0p0),xi)
                    if(bc):
                        bc2=True
                    jv, bc = oct_bc_cu(float(jj+vp0p0),yi)
                    if(bc):
                        bc3=True
                    iv1 = int(iv)
                    jv1 = int(jv)
                    if(iv1 == xi-1):
                        iv1= xi-2
                    if(jv1 == yi-1):
                        jv1 = yi-2

                    xitjv1 = xi*jv1
                    for c in range(0,nchan):
                        xityitc = xityi*c
                        lxyz3d = ii+xi*jj+xityitc

                        c1 = iv1+xitjv1+xityitc
                        c2 = c1+1 
                        c3 = c1+xi 
                        c4 = c3+1
                        # //This step could be done every K iteration alternatively to improve computational efficiency
                        g2,p1,p2,p3,p4 = oct_binterp_coefs_cu(iv,jv,iv1,iv1+1,jv1,jv1+1,geo2[c1],geo2[c2],geo2[c3],geo2[c4]) 
                        Ix  = oct_coef_binterp_cu(p1,p2,p3,p4,gradxarrg2[c1],gradxarrg2[c2],gradxarrg2[c3],gradxarrg2[c4])
                        Iy  = oct_coef_binterp_cu(p1,p2,p3,p4,gradyarrg2[c1],gradyarrg2[c2],gradyarrg2[c3],gradyarrg2[c4])
                        Ixx = oct_coef_binterp_cu(p1,p2,p3,p4,gradxxarr[c1], gradxxarr[c2],gradxxarr[c3],gradxxarr[c4])
                        Ixy = oct_coef_binterp_cu(p1,p2,p3,p4,gradxyarr[c1], gradxyarr[c2],gradxyarr[c3],gradxyarr[c4])
                        Iyy = oct_coef_binterp_cu(p1,p2,p3,p4,gradyyarr[c1], gradyyarr[c2],gradyyarr[c3],gradyyarr[c4])
                        # //Boundary condition gradient value settings
                        if(bc2):
                            Ix=0.
                            Ixx=0.
                            Ixy=0.
                        if(bc3):
                            Iy=0.
                            Ixy=0.
                            Iyy=0.
                        # //Here are the image derivatives needed for Brox variational optical flow
                        It     = g2-geo1[int(lxyz3d)]
                        Ixt    = Ix-gradxarr[int(lxyz3d)]
                        Iyt    = Iy-gradyarr[int(lxyz3d)]
                        yin    = It
                        yinx   = Ixt
                        yiny   = Iyt
                        IxIx   = Ix*Ix
                        IyIy   = Iy*Iy
                        IxxIxx = Ixx*Ixx
                        IxyIxy = Ixy*Ixy
                        IyyIyy = Iyy*Iyy
                        # //Zimmer normalization constants
                        if(dozim):
                            na = 1./(IxIx + IyIy+1.)
                            nb = 1./(IxxIxx+IxyIxy+1.)
                            nc = 1./(IxyIxy+IyyIyy+1.)
                        else:
                            na = 1.; nb = 1.; nc = 1.
                        intcomp  += na*yin*yin
                        intcomp2 += (nb*yinx*yinx+nc*yiny*yiny)

                        vr1  += (na*IxIx)
                        vr12 += (nb*IxxIxx+nc*IxyIxy)

                        IxIy   = na*Ix*Iy
                        IxxIxy = nb*Ixx*Ixy
                        IxyIyy = nc*Iyy*Ixy
                        LaVal  = (IxxIxy+IxyIyy)
                        vr2  += (IxIy)
                        vr22 += LaVal
                        vr3  += (IxIy)
                        vr32 += (LaVal)
                        vr4  += (na*IyIy)
                        vr42 += ((nb*IxyIxy+nc*IyyIyy))
                        natIt = -na*It
                        nbtIxt = nb*Ixt
                        nctIyt = nc*Iyt
                        vr5  += natIt*Ix
                        vr52 += -(nbtIxt*Ixx+nctIyt*Ixy)
                        vr6  += natIt*Iy
                        vr62 += -(nbtIxt*Ixy+nctIyt*Iyy)
                    # //with the above calculated, we are ready to compute individual terms in the big sparse matrix
                    psid = oct_PSI_data_cu(intcomp,0)/alpha #//Note for ease of computation I am combining psid and alpha here
                    psidq = 1./alpha
                    psid2 = lambdadalpha*oct_PSI_data_cu(intcomp2,0)
                    psidq2 = lambdadalpha


                    a1 = float((al1)*(psidq*(vr1)+psidq2*(vr12)+lambdac + psistotq)+(1-al1)*(psid*(vr1)+psid2*vr12+lambdac + psistot))
                    a2 = float((al1)*(psidq*(vr2)+psidq2*vr22)+(1-al1)*(psid*(vr2)+psid2*vr22))
                    a3 = float((al1)*(psidq*(vr3)+psidq2*vr32)+(1-al1)*(psid*(vr3)+psid2*vr32))
                    a4 = float((al1)*(psidq*(vr4)+psidq2*vr42+lambdac + psistotq)+(1-al1)*(psid*(vr4)+psid2*vr42+lambdac + psistot))
                    # //smoothness constraint terms in the sparse matrix
                    a5 = float(-1*(al1+(1-al1)*(psis1)))
                    # //Terms multiplied by du and dv at i, j-1 in the du and dv equation
                    a6 = float(-1*(al1+(1-al1)*(psis2)))

                    # //Terms multipled by du and dv at i+1, j in the du and dv equation
                    a7 = float(-1*(al1+(1-al1)*(psis3)))
                    # //Terms multiplied by du and dv at i, j+1 in the du and dv equation
                    a8 = float(-1*(al1+(1-al1)*(psis4))) 
                    # With the terms set, it is time to find rowt and rowu
                    rowt = 0
                    
                    if(jj > 0):
                        rowt += lxyza-xi

                    if(ii == 0):
                        rowt += lxyza-jj
                    else:
                        rowt += lxyza-jj-1

                    # //count a1 and a2 inserts here
                    rowt += lxyza+lxyza

                    # //counts every i < xi-1
                    rowt += lxyza-jj
                    # //counts every j less than yi-1
                    if(jj < yi-1):
                        rowt += lxyza
                    else:
                        rowt += xityi-xi
                    if(jj > 0):
                        rowt += lxyza-xi
                    if(ii == 0):
                        rowt += lxyza-jj; 
                    else:
                        rowt += lxyza-jj-1
                    rowt += lxyza+lxyza
                    
                    rowt += lxyza-jj
                    

                    if(jj < yi-1):
                        rowt += lxyza
                    else:
                        rowt += xityi-xi
                    rowu = lxyz
                    lxyz1 = lxyz-2 #//This is supposed to be to the adjacent x
                    lxyz2 = lxyz-xi2 #//This is supposed to be to the adjacent y
                    lxyz3 = lxyz+2 #//This is supposed to be to the adjacent x
                    lxyz4 = lxyz+xi2 #//This is supposed to be to the adjacent y
                    if(ii == 0):
                        lxyz1 += 4
                    if(jj == 0):
                        lxyz2 += (xi2+xi2)
                    if(ii == xi-1):
                        lxyz3 = lxyz3-4
                    if(jj == yi-1):
                        lxyz4 = lxyz4-(xi2+xi2)
                    # //OK, now time to actually fill the sparse matrices Aval and Mval
                    rowset = 0
                    if(jj > 0):
                        if(jj<yi-1):
                            Aval[int(rowt)]=a6
                        else:
                            Aval[int(rowt)]=a6+a8
                        Arow[int(rowt)] = lxyz
                        Acol[int(rowt)] = lxyz2
                        ArowSP[rowu]=rowt
                        rowu = rowu+1
                        rowset = 1
                        rowt = rowt+1

                    if(ii > 0):
                        if(ii<xi-1):
                            Aval[int(rowt)]=a5
                        else:
                            Aval[int(rowt)]=a5+a7
                        Arow[int(rowt)] = lxyz
                        Acol[int(rowt)] = lxyz1
                        if(rowset == 0):
                            ArowSP[int(rowu)]=rowt
                            rowu = rowu+1
                            rowset = 1
                        rowt = rowt+1

                    Aval[int(rowt)]=a1
                    Arow[int(rowt)] = lxyz
                    Acol[int(rowt)] = lxyz
                    if(rowset ==0):
                        ArowSP[int(rowu)]=rowt
                        rowu = rowu+1
                    Mval[int(lxyz)] = Aval[int(rowt)]
                    Mrow[int(lxyz)] = Arow[int(rowt)]
                    rowt = rowt+1
                    Aval[int(rowt)]=a2
                    Arow[int(rowt)] = lxyz
                    Acol[int(rowt)] = lxyz+1
                    rowt = rowt+1
                    if(ii < xi-1):
                        if(ii > 0):
                            Aval[int(rowt)]=a7
                        else:
                            Aval[int(rowt)]=a7+a5
                        Arow[int(rowt)] = lxyz
                        Acol[int(rowt)] = lxyz3
                        rowt = rowt+1

                    if(jj < yi-1):
                        if(jj > 0):
                            Aval[int(rowt)]=a8
                        else:
                            Aval[int(rowt)]=a8+a6
                        Arow[int(rowt)] = lxyz
                        Acol[int(rowt)] = lxyz4
                        rowt = rowt+1
                    rowset = 0
                    if(jj > 0):
                        if(jj<yi-1):
                            Aval[int(rowt)]=a6
                        else:
                            Aval[int(rowt)]=a6+a8
                        Arow[int(rowt)] = lxyz+1
                        Acol[int(rowt)] = lxyz2+1
                        ArowSP[int(rowu)]=rowt
                        rowu = rowu+1
                        rowset = 1
                        rowt = rowt+1
                    if(ii > 0):
                        if(ii<xi-1):
                            Aval[int(rowt)]=a5
                        else:
                            Aval[int(rowt)]=a5+a7
                        Arow[int(rowt)] = lxyz+1
                        Acol[int(rowt)] = lxyz1+1
                        if(rowset == 0):
                            ArowSP[int(rowu)]=rowt
                            rowu = rowu+1
                            rowset = 1
                        rowt = rowt+1
                    Aval[int(rowt)]=a3
                    Arow[int(rowt)] = lxyz+1
                    Acol[int(rowt)] = lxyz
                    if(rowset == 0):
                        ArowSP[int(rowu)]=rowt
                        rowu = rowu+1
                    rowt = rowt+1
                    Aval[int(rowt)]=a4
                    Arow[int(rowt)] = lxyz+1
                    Acol[int(rowt)] = lxyz+1
                    Mval[lxyz+1] = Aval[int(rowt)]
                    Mrow[lxyz+1] = Arow[int(rowt)]
                    rowt = rowt+1
                    if(ii < xi-1):
                        if(ii > 0):
                            Aval[int(rowt)]=a7
                        else:
                            Aval[int(rowt)]=a7+a5
                        Arow[int(rowt)] = lxyz+1
                        Acol[int(rowt)] = lxyz3+1
                        rowt = rowt+1
                    if(jj < yi-1):
                        if(jj > 0):
                            Aval[int(rowt)]=a8
                        else:
                            Aval[int(rowt)]=a8+a6
                        Arow[int(rowt)] = lxyz+1
                        Acol[int(rowt)] = lxyz4+1
                        rowt = rowt+1
                    #Also fill the vector bcu, which will then be used to solve for x.
                    val2 = lambdac*(uval[lxyza]-uvalt[lxyza])
                    bcu[lxyz] = float(al1*(psidq*(vr5)+psidq2*vr52-val2+psisnmiuq-psistotq*uval[lxyza])+
                       (1.-al1)*(psid*(vr5)+psid2*vr52-val2+psisnmiu-psistot*uval[lxyza]))
                    val2 = lambdac*(vval[lxyza]-vvalt[lxyza])  #//This is new I guess
                    bcu[lxyz+1]=float(al1*(psidq*(vr6)+psidq2*vr62-val2+psisnmivq-psistotq*vval[lxyza])+
                        (1-al1)*(psid*(vr6)+psid2*vr62-val2+psisnmiv-psistot*vval[lxyza]))
                
                
                grid.sync()
                #with the sparse matrix Aval and vector bcu, time to solve for Aval * x = bcu

                jMatXVec(Aval,Arow,ArowSP, Acol,x0,An,nx,dummyvec)
                grid.sync()

                gtr = cuda.grid(1)
                gs = cuda.gridsize(1)
                for i in range(gtr,nx,gs):
                    bcu[i]= bcu[i] - dummyvec[i]
                grid.sync()
                jDiagInv(Mval,nx) 
                grid.sync()
                jMatXVec(Mval,Mrow,Mrow,Mrow,bcu,nx,nx,z0) 
                grid.sync()
                gtr = cuda.grid(1)
                gs = cuda.gridsize(1)
                for j in range(gtr,nx,gs):
                    p0[j] = z0[j]
                grid.sync()

                if((cuda.threadIdx.x == 0) and (cuda.blockIdx.x == 0)):
                    residc[0] = 0.0
                jVecXVec(bcu,bcu,residc,nx) #,cta,grid)
                grid.sync()

                ki = 0

                while(((residc[0]) > tol) and (ki < iters)):
                    if(ki > 0):
                        if((cuda.threadIdx.x == 0) and (cuda.blockIdx.x == 0)):
                            z0tr0[0] = 0.0
                        jVecXVec(z0,bcu,z0tr0,nx) #,cta,grid) #; //Should be floats
                        grid.sync()

                        jMatXVec(Mval,Mrow,Mrow,Mrow,rk,nx,nx,z0) #; ///z0 updated here!!!
                        grid.sync()
                        if((cuda.threadIdx.x == 0) and (cuda.blockIdx.x == 0)):
                            zktrk[0] = 0.0
                        jVecXVec(z0,rk,zktrk,nx) #,cta,grid)
                        grid.sync()
                        Bk = (zktrk[0]) / (z0tr0[0])
                        jVecPVec(p0,z0,p0,Bk,nx) #; //!!!p0 updated here!!!!
                        grid.sync()

                        # //Now that this is all done, reset bcu and loop again
                        gtr = cuda.grid(1)
                        gs = cuda.gridsize(1)
                        for j in range(gtr,nx,gs):
                            bcu[j] = rk[j]
                        grid.sync()
                    if((cuda.threadIdx.x == 0) and (cuda.blockIdx.x == 0)):
                        rkTzk[0] = 0.0
                    jVecXVec(bcu,z0,rkTzk,nx) #,cta,grid)
                    grid.sync()


                    jMatXVec(Aval,Arow,ArowSP,Acol,p0,An,nx,dummyvec) #; //replacing pkTA with Atx0 for memory
                    grid.sync()
                    if((cuda.threadIdx.x == 0) and (cuda.blockIdx.x == 0)):
                        pkTApk[0] = 0.0
                    jVecXVec(p0,dummyvec,pkTApk,nx) #,cta,grid)
                    grid.sync()

                    
                    alphak = (rkTzk[0]) / (pkTApk[0])
                    jMatXVec(Aval,Arow,ArowSP,Acol, p0,An,nx,dummyvec)
                    grid.sync()
                    jVecPVec(p0,x0,x0,alphak,nx)
                    grid.sync()
                    jVecPVec(dummyvec,bcu,rk,-1.*alphak,nx)
                    grid.sync()
                    if((cuda.threadIdx.x == 0) and (cuda.blockIdx.x == 0)):
                        residc[0] = 0.0
                    jVecXVec(rk,rk,residc,nx) #,cta,grid)

                    grid.sync()
                    ki = ki+1
                grid.sync()
                gtr = cuda.grid(1)
                gs = cuda.gridsize(1)
                # //At this point, x0 is solved, so update the u/v vectors accordingly
                for kii in range(gtr,xityi,gs):
                    kii2 = kii+kii
                    uval[kii] = uval[kii]+x0[kii2]
                    vval[kii] = vval[kii]+x0[kii2+1] 
                    x0[kii2] = 0. 
                    bcu[kii2] = 0.
                    x0[kii2+1] = 0.
                    bcu[kii2+1] = 0.
                grid.sync()
            #endfor liters (inner iterations)
        #endfor GNC (Graduated Non-Convexity iterations)
        grid.sync()
        # re-use u/vvalt arrays to save on storage and pass flow values that need to be zoomed in
        gtr = cuda.grid(1)
        gs = cuda.gridsize(1)
        for lxyza in range(gtr,xityi,gs):
            uvalt[lxyza] = uval[lxyza] #note that uval/vval store the current iteration of optical flow
            vvalt[lxyza] = vval[lxyza]
        grid.sync()
        xio = xi #; //set the previous resolution values
        yio = yi
        grid.sync()

### End GPU Kernels
#This is the call to octane on python
# VARIABLES:
# f1, f2 are image1 and image 2 to compute optical flow with
# nchan = number of channels
# alpha = smoothness contraint strength
# lambdav = gradient constraint strength
# kiters = outer iterations/number of pyramid levels
# liters = inner iterations (number of updates to the linearized robust function derivatives)
# iters = number of conjugate gradient iterations
# blockspergrid/threadsperblock = CUDA GPU settings
# device = CUDA GPU device # to use
# imagemin/max = normalization min max which converts the image to float values of 0-255
#   Note that OCTOPY crops any values below imagemin, and sets to 0 (to accomodate missing values if needed)
# dozim = use the Zimmer et al. 2011 BC constraint normalization factors
# high_precision (bool), when set to true, will set all dtypes to double (warning, significantly increases GPU mem usage)
#   when set to false (default), all dtypes will be float32
# RETURNS:
# u/v displacements from input image sequence, assumes default x dimension is column motion, y dimension is row motion
# set cimage=False to flip this behavior
def octopy(f1, f2, nchan=1, alpha=3, lambdav=1, kiters=4,liters=3, iters=30,
           blockspergrid=125,threadsperblock=128,device=0,imagemin='none',imagemax='none',dozim=True,high_precision=False,cimage=True):
    if(high_precision):
        of_dtype = np.float64
        of_itype = np.int64
    else:
        of_dtype = np.float32
        of_itype = np.int32
    if((imagemin=='none') | (imagemax=='none')):
        print("Warning, imagemin/max not set, defaulting to f1 min/max")
        imagemin=np.amin(f1)
        imagemax=np.amax(f1)
    lambdadalpha = lambdav/alpha #args.lambda/alpha;
    nx, ny = f1.shape
    xityi = nx*ny
    xityi2 = xityi*2
    An=12*xityi-4*nx-4*ny

    lambdac = 0./alpha #args.lambdac/alpha;
    scsig = float(1.)
    
    scaleFactor= 0.5 #(float) args.scaleF;
    factor = scaleFactor**(kiters-1) #pow(scaleFactor,kiters-1);
    sigma = 0.6*sqrt(1.0/(factor*factor)-1.0)
    filtsize = int(2*sigma)
    if(filtsize < 5):
        filtsize = 5 #; //min gauss filter size
    filtsize2 = 2*filtsize+1
    cuda.select_device(device)
    # //Future iterations of OCTANE may have error checking on the cuda functions below -J. Apke 2/21/2022

    GK = np.zeros((filtsize2),dtype=of_dtype)
    Aval = np.zeros((An),dtype=of_dtype) #
    Arow = np.zeros((An),dtype=of_itype) #
    Acol = np.zeros((An),dtype=of_itype) #
    uval = np.zeros((xityi),dtype=of_dtype)
    vval = np.zeros((xityi),dtype=of_dtype)
    uvalt = np.zeros((xityi),dtype=of_dtype)
    vvalt = np.zeros((xityi),dtype=of_dtype)
    uhval = np.zeros((xityi),dtype=of_dtype)
    uhval = np.zeros((xityi),dtype=of_dtype)

    geo1 = np.zeros((xityi*nchan),dtype=of_dtype)
    geo2 = np.zeros((xityi*nchan),dtype=of_dtype)
    geo10 = np.zeros((xityi*nchan),dtype=of_dtype)
    geo20 = np.zeros((xityi*nchan),dtype=of_dtype)
    gradxarrg2 = np.zeros((xityi*nchan),dtype=of_dtype)
    gradyarrg2 = np.zeros((xityi*nchan),dtype=of_dtype)
    gradxxarr = np.zeros((xityi*nchan),dtype=of_dtype)
    gradxyarr = np.zeros((xityi*nchan),dtype=of_dtype)
    gradyyarr = np.zeros((xityi*nchan),dtype=of_dtype)
    gradxarr = np.zeros((xityi*nchan),dtype=of_dtype)
    gradyarr = np.zeros((xityi*nchan),dtype=of_dtype)


    ArowSP = np.zeros((xityi2),dtype=of_itype) #
    bcu = np.zeros((xityi2),dtype=of_dtype)
    x0 = np.zeros((xityi2),dtype=of_dtype)
    p0 = np.zeros((xityi2),dtype=of_dtype)
    z0 = np.zeros((xityi2),dtype=of_dtype)
    rk = np.zeros((xityi2),dtype=of_dtype)
    Mval = np.zeros((xityi2),dtype=of_dtype)
    Mrow = np.zeros((xityi2),dtype=of_itype)
    CTHc = 0
    CTHo = 0
    if(nchan < 2):
        dummyvec = np.zeros((xityi2),dtype=of_dtype)
    else:
        dummyvec = np.zeros((xityi*nchan),dtype=of_dtype)
    rkTzk = np.zeros(1).astype(of_dtype)
    pkTApk = np.zeros(1).astype(of_dtype) 
    zktrk = np.zeros(1).astype(of_dtype) 
    z0tr0 = np.zeros(1).astype(of_dtype) 
    residc = np.zeros(1).astype(of_dtype) 
    uarr = np.zeros((nx,ny)).flatten(order='F')
    varr = np.zeros((nx,ny)).flatten(order='F')
    uval = np.copy(uarr).astype(of_dtype)
    vval = np.copy(varr).astype(of_dtype)
    uhval = np.copy(uarr).astype(of_dtype)
    vhval = np.copy(varr).astype(of_dtype)

    x0[:] = 0


    minout = 0.
    maxout = 255.
    maxin = imagemax
    minin = imagemin

    
    geof1data = ((f1-float(minin))/(maxin-minin))*((maxout-minout)+minout)
    geof2data = ((f2-float(minin))/(maxin-minin))*((maxout-minout)+minout)
    #for geoips, add this (missing values will be below 0)
    cond1 = geof1data < 0
    geof1data[cond1] = 0.
    cond1 = geof2data < 0
    geof2data[cond1] = 0.
    

    image1 = (geof1data).flatten(order='F')
    image2 = (geof2data).flatten(order='F') 
    geo1 = np.copy(image1).astype(of_dtype)
    geo2 = np.copy(image2).astype(of_dtype)
    geo10 = np.copy(image1).astype(of_dtype)
    geo20 = np.copy(image2).astype(of_dtype)

    tol = 0.0001*0.0001 #conjugate gradient convergence tolerance

    #call the kernel here
    print("Optical flow code starting")
    t0 = time.time()
    octane[blockspergrid, threadsperblock](Aval,Arow,ArowSP,Acol,x0,
        uval,vval,uvalt,vvalt,uhval,vhval,
        geo1,geo2,geo10,geo20,CTHc,CTHo,GK,gradxarrg2,
        gradyarrg2,gradxxarr,gradxyarr,gradyyarr,
        gradxarr,gradyarr,nchan,alpha,lambdadalpha,lambdac,
        An,xityi2,nx,ny,
        bcu,Mval,Mrow,z0,
        p0,residc,tol,iters,rk,zktrk,z0tr0,
        rkTzk,pkTApk,dummyvec,
        liters,kiters,filtsize,filtsize2,scaleFactor,scsig,dozim)
    uval = np.asarray(uval.reshape((nx,ny),order='F'))
    vval = np.asarray(vval.reshape((nx,ny),order='F'))
    t1 = time.time()
    total = t1-t0
    print("Time to complete: ",total)
    if(cimage):
        return vval, uval
    else:
        return uval, vval


