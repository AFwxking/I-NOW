#Name: jma_array_binterp.py
#Purpose: A bilinear and nearest neighbor interpolation function that can interpolate
#   2d arrays of x/y points if needed
#Inputs:
# arr: array to interpolate from
# x/y: set of x/y points in the array space to interpolate from, can be floats
#Returns:
# array of the same shape as x and y inputs, containing interpolated values

import numpy as np

def jma_array_binterp2d(arr,x,y,dotwod=0,maskneg=0):
    x1 = x.astype(int)
    y1 = y.astype(int)
    nxy = arr.shape
    nx = nxy[0]
    ny = nxy[1]
    x1 = np.clip(x1,0,nx-2)
    y1 = np.clip(y1,0,ny-2)
    x2 = x1+1
    y2 = y1+1


    v1 = ((x2-x)/(x2-x1))
    v2 = ((x-x1)/(x2-x1))
    if(arr.ndim > 2):
        f3 = np.copy(arr)
        nz = nxy[2]

        for i in range(0,nz):
            f1 = v1*arr[x1,y1,i]+v2*arr[x2,y1,i]
            f2 = v1*arr[x1,y2,i]+v2*arr[x2,y2,i]

            f3[:,:,i] = ((y2-y)/(y2-y1))*f1+((y-y1)/(y2-y1))*f2
    
    else:
        f1 = v1*arr[x1,y1]+v2*arr[x2,y1]
        f2 = v1*arr[x1,y2]+v2*arr[x2,y2]

        f3 = ((y2-y)/(y2-y1))*f1+((y-y1)/(y2-y1))*f2

    if(maskneg == 0):
        return f3
    else:
        cond1 = ((arr[x1,y1] < 0) | (arr[x2,y1] < 0) | (arr[x1,y2] < 0) | (arr[x2,y2] < 0))
        f3[cond1] = -1 
        return f3
def jma_array_nn(arr,x,y,dotwod=0,maskneg=0):
    x1 = np.round(x).astype(int)
    y1 = np.round(y).astype(int)
    nxy = arr.shape
    nx = nxy[0]
    ny = nxy[1]
    x1 = np.clip(x1,0,nx-2)
    y1 = np.clip(y1,0,ny-2)
    if(arr.ndim > 2):
        f3 = np.copy(arr)
        nz = nxy[2]

        for i in range(0,nz):

            f3[:,:,i] = arr[x1,y1,i] 
    
    else:

        f3 = arr[x1,y1] 

    if(maskneg == 0):
        return f3
    else:
        cond1 = ((arr[x1,y1] < 0) | (arr[x2,y1] < 0) | (arr[x1,y2] < 0) | (arr[x2,y2] < 0))
        f3[cond1] = -1 
        return f3

