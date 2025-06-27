# 

import numpy as np
from jma_goesread import *
from octopy import *
from jma_array_binterp import *
from octane_temporal_interpolation import *

def INOW_nowcaster(file1,file2,time_to_interp, set_scalar=False, scalar='none', settime = False, incr_frames=1, alpha=3, lambdav=.3, normmin=(-1.6443),normmax = (4094*0.04572892-1.6443), device = 0, warp = True):
    '''Updated oct_ti code that does optical flow nowcast on scalar value rather than input for optical flow (i.e. Brightness Temperature)
    
    Inputs: 
    1) file1: directory for first image
    2) file2: directory for second (later) image
    3) time_to_interp: datetime object for target time of nowcast

    Outputs:

    '''
    
    #Reading raw data from GOES files
    g1 = jma_goesread(file1,cal='RAW')
    g2 = jma_goesread(file2,cal='RAW')
    
    #Getting start times of both files and end time of first
    start_time = g1.timestart 
    start_time2 = g2.timestart 
    end_time = g1.timeend
    
    #Converting masked values to 0
    g1.data[g1.data.mask] = 0.
    g2.data[g2.data.mask] = 0.
    
    #Time in seconds between first file end time and first file start time
    time_dt = (end_time-start_time).total_seconds()

    #Image data for optical flow calculation
    im1 = np.asarray(g1.data) 
    im2 = np.asarray(g2.data) 

    #Optical flow calculation
    uvaltest, vvaltest = octopy(im1,im2,imagemin=normmin,imagemax=normmax,alpha=alpha,lambdav=lambdav, device=device)

    #Creating meshgrid based on size of first input file
    sx,sy = im1.shape 
    yarr, xarr = np.meshgrid(np.arange(0,sy),np.arange(0,sx))
    
    #Calculating time difference for scans
    tslope = time_dt/(sx-1)
    timearr = xarr*tslope
    if (settime == False):
        timearr[:,:] = 0.
    
    #Reading Brightness Temp
    g1bt = jma_goesread(file1,cal='TEMP')
    g2bt = jma_goesread(file2,cal='TEMP')
    im1bt = g1bt.data.data
    im2bt = g2bt.data.data
    
    #If option for scalar is set to True
    if(set_scalar):
        #Check if im1bto is still a string
        if(isinstance(scalar,str)):
            print("warning, set the scalar variable, defaulting to brightness temp")
            scalar = im1bt
    else:
        scalar = im1bt

    #Calculating time in seconds for each increment
    if time_to_interp.tzinfo is None:
        print("The datetime object is offset-naive...continuing")
    else:
        print("The datetime object is offset-aware...updating to offset-naive.")
        time_to_interp = time_to_interp.replace(tzinfo=None)
    time_dt2 = ((time_to_interp-start_time).total_seconds())/(incr_frames)

    # Setting dt3 to 0
    time_dt3 = 0 

    #If scalar is 3 dimensions...using CTH/CBH
    if scalar.ndim == 3:
        scalar2 = np.copy(scalar[:,:,1])
        scalar2[:,:] = 0
        con1=scalar[:,:,1] >=0
        scalar2[con1] = (scalar[:,:,1])[con1]
    #If scalar is 2 dimensions...using just brightness temp
    else:
        scalar2 = np.copy(scalar)
        scalar2[:,:] = 0
        con1=scalar[:,:] >=0
        scalar2[con1] = (scalar[:,:])[con1]

    #Preallocate array for scalar values
    scalar_array = np.empty((incr_frames, *np.shape(scalar)))

    for j in range(0,incr_frames):

        print(f'Incrementing here {j} out of {incr_frames}.')

        #Adding to dt3 with time_dt2
        time_dt3 += time_dt2 

        #Current target time for extrapolation
        testtime = (time_dt2 - timearr)/(start_time2-start_time).total_seconds()
        testtime2 = (time_dt3 - timearr)/(start_time2-start_time).total_seconds() 

        #Interpolation...moving u and v vectors to next increment
        if warp == True:
            
            if set_scalar: #If nowcasting CTH/CBH use CTH for occlusion reasoning
                ut, vt = octane_temporal_interpolation((50000 - scalar2), im2bt,uvaltest,vvaltest,testtime,donowcast=True)
            else: #If nowcasting brightness temp, use brightness temp for occlusion reasoning
                ut, vt = octane_temporal_interpolation(scalar2, im2bt,uvaltest,vvaltest,testtime,donowcast=True)

            #for each increment, ut and vt take over for uvaltest and vvaltest
            uvaltest = np.copy(ut)
            vvaltest = np.copy(vt)
        else: #warp is false so not warping optical flow field, just using original optical flow
            ut = np.copy(uvaltest)
            vt = np.copy(vvaltest)

        #New placement of scalar values
        utx0 = xarr-vt*testtime
        vtx0 = yarr-ut*testtime    
        utx0_2 = xarr-vt*testtime2
        vtx0_2 = yarr-ut*testtime2
        if set_scalar: 
            I0X0 = jma_array_nn(scalar,utx0_2,vtx0_2) #Using nearest neighbor interpolation for CTH/CBH Nowcast
        else:
            I0X0 = jma_array_binterp2d(scalar,utx0_2,vtx0_2) #Using bilinear interpolation for brightness temperature nowcast
            print('Using bilinear interpolation')

        #New placement of first brightness temperature
        I0X0_im1 = jma_array_binterp2d(scalar2,utx0,vtx0)
        
        #Making copies of scalar value and first input brightness temperature input
        scalar2 = np.copy(I0X0_im1)

        
        #If more than one increment
        if incr_frames > 1:
            if scalar_array.ndim == 4:
                scalar_array[j, :, :, :] = I0X0
            elif scalar_array.ndim == 3:
                scalar_array[j, :, :] = I0X0
            else:
                print(f'Dimensions are not compatible for function...check number of dims for scalar.')
                exit()
        else:
            scalar_array = np.copy(I0X0)

    return scalar_array
