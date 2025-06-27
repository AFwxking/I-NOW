###
#Purpose:  This code is designed to read and navigate GOES-R netcdf4s
#Author: Jason M. Apke 
#Date: 9/11/2017
#Inputs:  a GOES-R netcdf file full path (string), 
#   cal- calibration preference string, RAW, BRIT, TEMP, or REF
#   nav- True/False navigate the file (default False)
#Requires: netcdf4, math, numpy libraries
#Outputs: An object containing the following:
#   data- nx x ny 2d numpy array with the calibrated image values
#   lat - nx x ny 2d latitude array (if nav == 1), else 0
#   lon - nx x ny 2d longitude array (if nav == 1), else 0
#   x/y - nx x ny 2d array of i/j values for navigation
#   x/y offset- offset value for converting x/y to mirror scan angles (in radians)
#   x/yscale- scale value for converting x/y to mirror scan angles
#   tvs - Time value string from satellite file (center scan time), in YYYYmmdd-HHMMSS format
#   timestart - Time value string from satellite file (start scan time), in YYYYmmdd-HHMMSS format
#   timeend - Time value string from satellite file (end scan time), in YYYYmmdd-HHMMSS format
#   pph - (navigation variable) point-perspective height
#   req - (navigation variable) semi-major axis (radius of earth at the equator)
#   rpol- (navigation variable) semi-minor axis (radius of earth at the pole)
#   lam0- (navigation variable) longitude of the satellite
#   H   - (navigation variable) satellite distance from Earth's center (pph+req)

from math import *
import netCDF4
import numpy as np
import datetime

class jma_goesread(object):
    def __init__(self,path,cal="RAW",nav=False):
        #check capitals
        if cal == "Raw" or cal=="raw":
            cal = "RAW"
        if cal == "Brit" or cal=="brit":
            cal = "BRIT"
        if cal == "Temp" or cal=="temp":
            cal = "TEMP"
        if cal == "Ref" or cal=="ref":
            cal = "REF"
        if cal != "RAW" and cal != "BRIT" and cal != "TEMP" and cal != "REF":
            print("Cal incorrectly set, use raw, brit, temp or ref, setting to raw")
            cal = "RAW"

        #path = '/mnt/grb/goes16/2018/2018_07_28_209/abi/L1b/RadM1/OR_ABI-L1b-RadM1-M3C02_G16_s20182092100279_e20182092100337_c20182092100369.nc'
        nc = netCDF4.Dataset(path)
        data = np.squeeze(nc.variables['Rad'][:])
        x2 = np.squeeze(nc.variables['x'][:])
        y2 = np.squeeze(nc.variables['y'][:])
        xmin = x2[0]
        xmax = x2[len(x2)-1]
        ymax = y2[0]
        ymin = y2[len(y2)-1]
        xscale=nc.variables['x'].scale_factor
        xoffset=nc.variables['x'].add_offset
        yscale=nc.variables['y'].scale_factor
        yoffset=nc.variables['y'].add_offset
        x,y = np.meshgrid(x2,y2)
        band_wavelength = nc.variables['band_wavelength'][:]
        band_id = nc.variables['band_id'][:]
        
        longname= nc.variables['goes_imager_projection'].long_name
        req= nc.variables['goes_imager_projection'].semi_major_axis
        rpol= nc.variables['goes_imager_projection'].semi_minor_axis
        pph= nc.variables['goes_imager_projection'].perspective_point_height
        lam0= nc.variables['goes_imager_projection'].longitude_of_projection_origin
        sts = nc.time_coverage_start
        ste = nc.time_coverage_end
        if cal != "RAW":
            fk1= nc.variables['planck_fk1'][:]
            fk2= nc.variables['planck_fk2'][:]
            bc1= nc.variables['planck_bc1'][:]
            bc2= nc.variables['planck_bc2'][:]
            kap1= nc.variables['kappa0'][:]
        time_var = nc.variables['t']
        dtime = netCDF4.num2date(time_var[:],time_var.units)
        tvs = dtime.strftime('%Y%m%d-%H%M%S')
        tvs_start = datetime.datetime.strptime(sts[:-len('.8Z')],'%Y-%m-%dT%H:%M:%S')
        tvs_end = datetime.datetime.strptime(ste[:-len('.8Z')],'%Y-%m-%dT%H:%M:%S')
        lam0= radians(lam0)
        nc.close()
        H = pph+req
        
        if cal=="RAW":
            self.data=data
        if cal=="REF":
            self.data=kap1*data
        if cal=="TEMP":
            datacal = (fk2/(np.log((fk1/data)+1.))-bc1)/bc2
            self.data=datacal
        if cal=="BRIT":
            datacal = (fk2/(np.log((fk1/data)+1.))-bc1)/bc2
            datacal2 = fk1/(exp(fk2/(bc1+(bc2*datacal)))-1)
            self.data=datacal2

        #self.data= data #datacal
        self.x = x
        self.xscale = xscale
        self.xoffset = xoffset
        self.y = y
        self.yscale = yscale
        self.yoffset = yoffset
        self.tvs=tvs
        self.timestart=tvs_start
        self.timeend=tvs_end
        self.pph=pph
        self.req=req
        self.rpol=rpol
        self.lam0=lam0
        self.H = H
        self.band_wavelength = band_wavelength[0]
        self.band_id = band_id[0]
        #image bounds
        self.xmin = pph*xmin
        self.xmax = pph*xmax
        self.ymin = pph*ymin
        self.ymax = pph*ymax
        if(nav):
            print("Navigating file...")
            self.jma_goesnav() #fills lat/lon, separated so it can be run on subsets if needed
            print("...Done navigating")

    #Purpose: to navigate goes pixels, fills lat/lon from goes data objects, not recommended for large images
    #Input: geo- a GOESData object
    #outputs: adds lat/lon to the geo object, with option to add second, parallax corrected lat2/lon2 arrays
    def jma_goesnav(self, parallax_height=0.,nav2=False):
        a = (np.sin(self.x))**2.+((np.cos(self.x))**2.)*((np.cos(self.y))**2.+((self.req+parallax_height)**2.)/((self.rpol+parallax_height)**2.)*(np.sin(self.y))**2.)
        b = -2.* self.H*np.cos(self.x)*np.cos(self.y)
        c = self.H**2. - (self.req+parallax_height)**2.
        rs = (-b - (b**2. - 4.*a*c)**(0.5))/(2.*a)
        sx = rs*np.cos(self.x)*np.cos(self.y)
        sy = -rs*np.sin(self.x)
        sz = rs*np.cos(self.x)*np.sin(self.y)
        lat = np.arctan(((self.req+parallax_height)**2.)/((self.rpol+parallax_height)**2.)*(sz/((self.H-sx)**2. +sy**2.)**0.5))
        lon = self.lam0 - np.arctan(sy/(self.H-sx))
        if(nav2):
            self.lat2 = np.degrees(lat)
            self.lon2 = np.degrees(lon)
        else:
            self.lat = np.degrees(lat)
            self.lon = np.degrees(lon)

