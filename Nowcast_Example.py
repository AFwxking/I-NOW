# Filename: Nowcast_Example.py
# Purpose: Script to provide example code for I-NOW (Improved Nowcasting via Optical flow Warping) presented in "Nowcasting 
#   3D Cloud Fields Using Forward Warping Optical Flow" by Matthew King, Jason Apke, Steven Miller, Katherine Haynes, Yoo-Jeong Noh, 
#   and John Haynes, 2025: J. Atmos Ocea. Tech.
# DOI: TBD

# Requires: 
#   Python 3.12.2 (version used for development, other versions may work)
#   Data files which include CLAVRx_data and GOES Full Disk Band 13 data (code to download GOES data from AWS is included)
#   Python scripts contained in OCTOPY folder (contains scripts to run python version of OCTANE optical flow code and the I-NOW 
#       nowcast method which requires temporal interpolation scripts)
#   Code requires GPU with support for CUDA (developed on NVIDIA RTX A6000)
#   Data files within the DOI in data availability statement of manuscript (CLAVRx CTH/CBH data saved in clavrx_data folder on GitHub Repo)

# Dependencies: netcdf4, numba, cudatookit, s3fs, matplotlib, cartopy, xarray, metpy
# netcdf4, numba, and cudatoolkit required to run nowcasts
# s3fs required to download GOES data from AWS
# xarray, matplotlib, cartopy, and metpy required in code to plot results

# Recommend using a virtual environment to run this code, such as conda
# Development required install using the following commands:
    # create an anaconda environment:
        # $ conda create -n INOW_env python=3.12.2
        # $ conda activate INOW_env
    # Install Netcdf4 first, this one does not like to play nice with the others:
        # $ conda install -c conda-forge netcdf4
    # Install numba
        # $ conda install -c numba numba
    # Install the cudatoolkit, use the anaconda distribution, the conda-forge one has a driver issue
        # $ conda install -c anaconda cudatoolkit
    # Now matplotlib, use the pip-install for this
        # $ python -m pip install -U matplotlib
    # Now cartopy
        # $ conda install -c conda-forge cartopy
    # Now xarray
        # $ conda install -c conda-forge xarray
    # Now metpy
        # $ conda install -c conda-forge metpy
    # Now s3fs
        # $ conda install -c conda-forge s3fs

# environment.yml file provided in directory for easy installation of all dependencies
#   $ conda env create -f environment.yml

# Example run commnand (unix):
# $ python Nowcast_Example.py

# Code Author: Matthew P. King, Lt Col, USAF
# Date: 27 June 2025 
# Version: 1.0

# Explanation of code: 
# Nowcast_Example.py provides an example of how to run the I-NOW nowcasting method on GOES-16 Band 13 data.Code downloads GOES-16 
# Band 13 data from AWS, runs a python version of OCTANE optical flow code to calculate the optical flow using the values contained 
# in the JTEC article. Rather than linear extrapolation, the code uses a warping method formed from computer vision temporal 
# interpolation methods to account for occlusions and time related changes to the optical flow field in the forward time direction 
# rather than a time that is intermediate of two images. The code provides nowcasts of GOES-16 10.3 micron brightness temperature,
# CLAVR-x derived cloud top heights (CTH), and CLAVR-x derived cloud base heights (CBH). Observed brightness temperature data and 
# CLAVR-x data are plotted alongside the nowcast results in 10-minute intervals. The code also creates animated gifs of the nowcasts
# to easier visualize the results through time. The code is designed to be run on a GPU with CUDA support, and the user can define 
# the save directory for the results to be saved and data that is downloaded. CLAVR-x data is not downloadable from the internet,
# so the GitHub repository contains a folder with CLAVR-x data for the example date and time used in the code that has been reduced
# to contain only the required CTH and CBH data instead of full CLAVR-x retrieval data.  



#%%
# Imports
import os
import sys
sys.path.append('OCTOPY/')
from OF_Nowcaster import INOW_nowcaster
from custom_fns import get_sat_data_aws, get_clavrx_file_path, clavrx_truth, get_sat_parameters, gif_loop_maker
from jma_goesread import jma_goesread
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.plots import ctables

#%%
# Required inputs for nowcast functions

# Defining user save directory
save_dir = {'dir':''}  #Use '' for local directory, otherwise define a directory

# Determine base directory
base_dir = save_dir['dir'] if save_dir['dir'] else os.getcwd() #If falsy (i.e. ''), then defaults to current working directory

# #Values for nowcast
nowcast_extent = 180 #minutes (last valid time of nowcast)
time_interval = 10 #minutes (time interval between nowcast warping iterations)
sat_name = 'G16' #GOES-16, G17 = GOES-17, etc
date = '01/16/2023' #mm/dd/yyyy
time = '15:50' #hh:mm (UTC)
time2 = '16:00' #hh:mm (UTC)
scan = 'RadF' #RadF = Full disk, RadC = CONUS
device = 0 #Defines which GPU to use

# Define channel for GOES-16/17 Band 13 (10.3 micrometers)
channel = 13

# Load MetPy's IR colormap and normalization
cmap = ctables.registry.get_colortable('ir_rgbv')  # Metpy's 'ir_cloud' colormap for IR brightness temperatures

# Define channel for GOES-16/17 Band 13 (10.3 micrometers)
channel = 13

#Get sat data files (initial is time_interval behind initialization time; final is initialization time)
initial_file = get_sat_data_aws(sat_name, scan, date, time, save_dir['dir'], channel)
final_file = get_sat_data_aws(sat_name, scan, date, time2, save_dir['dir'], channel)

#I-NOW Specifics
initial_time = datetime.datetime.strptime(f'{date} {time2}', '%m/%d/%Y %H:%M') # Initial time for nowcast as datetime object
time_to_interp = initial_time + datetime.timedelta(minutes=nowcast_extent) # Time to which to extrapolate as datetime object
settime = False # If true, takes into account difference in actual time of scan; however, this will adversely affect validation with truth data
alpha=3 # alpha weight for brightness constancy
lambdav=.3 # lambda weight for smoothness constraint
warp=True # If true, warps optical flow with each time iteration; if false, only uses a linear extrapolation of the optical flow field
device=0 # GPU device
nowcast_iterations = int((nowcast_extent / time_interval)) #Defines nowcast iterations based on nowcast extent

#%%
#Run I-NOW (Brightness Temperature Nowcast)
Nowcast_array = INOW_nowcaster(final_file[0],initial_file[0],time_to_interp, set_scalar=False, 
        scalar='None', settime = settime, incr_frames=nowcast_iterations, alpha=alpha, 
        lambdav=lambdav, normmin=-1.6443, normmax = 4094*0.04572892-1.6443, device=device, warp = warp)

#%%
#Downloading observed sat brightness temperature data for truth data
truth_files = []
file_time = initial_time
while file_time <= time_to_interp:
    date_str = datetime.datetime.strftime(file_time, '%m/%d/%Y')
    time_str = datetime.datetime.strftime(file_time, '%H:%M')
    current_file = get_sat_data_aws(sat_name, scan, date_str, time_str, save_dir['dir'], channel)
    truth_files.append(current_file[0])
    file_time = file_time + datetime.timedelta(minutes=time_interval)

Truth_array = np.zeros((Nowcast_array.shape[0]+1, Nowcast_array.shape[1], Nowcast_array.shape[2]))
for truth_idx in range(Truth_array.shape[0]):
    
    # Reading Brightness Temp
    g1bt = jma_goesread(truth_files[truth_idx],cal='TEMP')

    if truth_idx == 0:
        g1bt.jma_goesnav()
        goes_lat = g1bt.lat
        goes_lon = g1bt.lon
    im1bt = g1bt.data.data
    data_mask = g1bt.data.mask
    im1bt[data_mask] = np.nan
    Truth_array[truth_idx, :, :] = im1bt.data
    if truth_idx >0:
        Nowcast_array[truth_idx - 1, data_mask] = np.nan
    print(f'Done appending truth file number {truth_idx}.')

#%%
#Plotting Brightness Temperature Nowcast Results and Truth Data

#Extract necessary data from satellite imagery file to make plots
sat_parameters = get_sat_parameters(final_file[0])

# Define the geostationary projection
satellite_longitude = sat_parameters['sat_lon']  # Replace with your satellite's central longitude
geo_proj = ccrs.Geostationary(central_longitude=satellite_longitude)

# Font size settings
title_fontsize = 16
label_fontsize = 12
colorbar_fontsize = 14
tick_fontsize = 10

# Plotting
file_time = initial_time
file_time_str = datetime.datetime.strftime(file_time,'%m/%d/%Y %H:%M UTC')
for idx in range(Nowcast_array.shape[0] + 1):
    print(f'Working on image {idx:02}.')

    # Define figure and GridSpec layout
    fig = plt.figure(figsize=(18, 10))
    spec = gridspec.GridSpec(ncols=2, nrows=2, height_ratios=[1, 0.05], hspace=0.1, wspace=0.05)

    # Create subplots for the images
    ax1 = fig.add_subplot(spec[0, 0], projection=geo_proj)
    ax2 = fig.add_subplot(spec[0, 1], projection=geo_proj)

    # Define extent based on data's spatial bounds
    extent = [
        sat_parameters['x_meters_min'],
        sat_parameters['x_meters_max'],
        sat_parameters['y_meters_min'],
        sat_parameters['y_meters_max']
    ]

    # Plot True Brightness Temperature (left)
    im1 = ax1.imshow(
        Truth_array[idx, :, :], cmap=cmap, origin='upper',
        vmin=160, vmax=330,
        extent=extent,
        transform=geo_proj
    )
    ax1.set_title(f'Observed GOES-16 10.3$\mu$m Tb [K]\nValid: {file_time_str}', fontsize=title_fontsize)
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax1.add_feature(cfeature.STATES, linewidth=0.5)
    gridlines = ax1.gridlines(draw_labels=True, crs=ccrs.PlateCarree())
    gridlines.right_labels = False
    gridlines.top_labels = False
    gridlines.xlabel_style = {'fontsize': label_fontsize}
    gridlines.ylabel_style = {'fontsize': label_fontsize}
    gridlines.xlabels_bottom = True
    gridlines.ylabels_left = True
    gridlines.xlabels_top = False
    gridlines.ylabels_right = False
    gridlines.xlabel_style.update({'fontsize': tick_fontsize})
    gridlines.ylabel_style.update({'fontsize': tick_fontsize})

    # Plot Nowcast Brightness Temperature (right)
    if idx == 0:
        im2 = ax2.imshow(
            Truth_array[idx, :, :], cmap=cmap, origin='upper',
            vmin=160, vmax=330,
            extent=extent,
            transform=geo_proj
        )
    else:
        im2 = ax2.imshow(
            Nowcast_array[idx - 1, :, :], cmap=cmap, origin='upper',
            vmin=160, vmax=330,
            extent=extent,
            transform=geo_proj
        )
    ax2.set_title(f'I-NOW ({idx*time_interval:02} min) 10.3$\mu$m Tb [K]\nValid: {file_time_str}', fontsize=title_fontsize)
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax2.add_feature(cfeature.STATES, linewidth=0.5)
    gridlines = ax2.gridlines(draw_labels=True, crs=ccrs.PlateCarree())
    gridlines.right_labels = False
    gridlines.top_labels = False
    gridlines.xlabel_style = {'fontsize': label_fontsize}
    gridlines.ylabel_style = {'fontsize': label_fontsize}
    gridlines.xlabels_bottom = True
    gridlines.ylabels_left = True
    gridlines.xlabels_top = False
    gridlines.ylabels_right = False
    gridlines.xlabel_style.update({'fontsize': tick_fontsize})
    gridlines.ylabel_style.update({'fontsize': tick_fontsize})

    # Add a single shared colorbar below the plots
    cbar_ax = fig.add_subplot(spec[1, :])  # Single subplot spanning both columns
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Brightness Temperature [K]', fontsize=colorbar_fontsize)
    cbar.ax.tick_params(labelsize=colorbar_fontsize)

    # Save the figure
    os.makedirs(f'{base_dir}/BT_Plots/', exist_ok=True)  # Ensure the directory exists
    plt.savefig(f'{base_dir}/BT_Plots/Nowcast{idx:02}.png', bbox_inches='tight')
    plt.close()

    # Update file_time for the next iteration
    file_time = file_time + datetime.timedelta(minutes=time_interval)
    file_time_str = datetime.datetime.strftime(file_time,'%m/%d/%Y %H:%M UTC')

#Make Loop
gif_loop_maker(f'{base_dir}/BT_Plots/', frame_duration = 300, loop_count = 0) #Frame duration in milliseconds, loop_count = 0 is infinite loop

#%%

#Run I-NOW (CTH/CBH Nowcast)

#Get CLAVRx_data
clavrx_file_path = get_clavrx_file_path(sat_name, scan, date, time2, save_dir['dir'])
CTH_values, CBH_values = clavrx_truth(clavrx_file_path[0][0])

#Predefine array to hold all extrapolated values
CBH_CTH_nowcast_array = np.empty( (nowcast_iterations + 1,CTH_values.shape[0], CTH_values.shape[1], 2 ) )
initial_CBH_CTH = np.empty((CTH_values.shape[0], CTH_values.shape[1], 2))

#Place initial conditions into arrays
CBH_CTH_nowcast_array[0, :, :, 0] = CBH_values
CBH_CTH_nowcast_array[0, :, :, 1] = CTH_values
initial_CBH_CTH[:,:,0] = np.copy(CBH_values)
initial_CBH_CTH[:,:,1] = np.copy(CTH_values)

#Run I-NOW for CTH/CBH Nowcast
CBH_CTH_nowcast_array[1:, :, :, :] = INOW_nowcaster(final_file[0],initial_file[0],time_to_interp, set_scalar=True, 
        scalar=initial_CBH_CTH, settime = settime, incr_frames=nowcast_iterations, alpha=alpha, 
        lambdav=lambdav, normmin=-1.6443, normmax = 4094*0.04572892-1.6443, device=device, warp = warp)

#%%
#Checking for truth CLAVR-x data
clavrx_truth_files = []
file_time = initial_time
while file_time <= time_to_interp:
    date_str = datetime.datetime.strftime(file_time, '%m/%d/%Y')
    time_str = datetime.datetime.strftime(file_time, '%H:%M')
    clavrx_file_path = get_clavrx_file_path(sat_name, scan, date_str, time_str, save_dir['dir'])
    clavrx_truth_files.append(clavrx_file_path[0][0])
    file_time = file_time + datetime.timedelta(minutes=time_interval)

CLAVRx_Truth_array = np.zeros((CBH_CTH_nowcast_array.shape[0], CBH_CTH_nowcast_array.shape[1], CBH_CTH_nowcast_array.shape[2], 2))
for clavrx_truth_idx in range(CLAVRx_Truth_array.shape[0]):
    
    # Reading CTH/CBH data
    CTH_values, CBH_values = clavrx_truth(clavrx_truth_files[clavrx_truth_idx])
    CLAVRx_Truth_array[clavrx_truth_idx, :, :, 0] = CBH_values
    CLAVRx_Truth_array[clavrx_truth_idx, :, :, 1] = CTH_values
    
    print(f'Done appending clavrx_truth file number {clavrx_truth_idx}.')

#%%
#Plotting CTH Nowcast Results and Truth Data

#Extract necessary data from satellite imagery file to make plots
sat_parameters = get_sat_parameters(final_file[0])

# Define the geostationary projection
satellite_longitude = sat_parameters['sat_lon']  # Replace with your satellite's central longitude
geo_proj = ccrs.Geostationary(central_longitude=satellite_longitude)

# Font size settings
title_fontsize = 16
label_fontsize = 12
colorbar_fontsize = 14
tick_fontsize = 10

# Setting colormap for CTH Plots (need to set 'none' for masked values)
cmap = plt.colormaps['viridis_r'].copy()
cmap.set_bad(color='none') 

# Plotting
file_time = initial_time
file_time_str = datetime.datetime.strftime(file_time,'%m/%d/%Y %H:%M UTC')
for idx in range(CBH_CTH_nowcast_array.shape[0]):
    print(f'Working on image {idx:02}.')

    # Define figure and GridSpec layout
    fig = plt.figure(figsize=(18, 10))
    spec = gridspec.GridSpec(ncols=2, nrows=2, height_ratios=[1, 0.05], hspace=0.1, wspace=0.05)

    # Create subplots for the images
    ax1 = fig.add_subplot(spec[0, 0], projection=geo_proj)
    ax2 = fig.add_subplot(spec[0, 1], projection=geo_proj)

    # Define extent based on data's spatial bounds
    extent = [
        sat_parameters['x_meters_min'],
        sat_parameters['x_meters_max'],
        sat_parameters['y_meters_min'],
        sat_parameters['y_meters_max']
    ]

    # Plot True CTH on left
    cth_data_truth = np.copy(CLAVRx_Truth_array[idx, :, :, 1])
    cth_data_truth = np.ma.masked_invalid(cth_data_truth)  # Mask NaN values for plotting
    im1 = ax1.imshow(
        cth_data_truth, cmap='viridis_r',
        origin='upper', vmin=0, vmax=15000,
        extent=extent, transform=geo_proj)

    ax1.set_title(f'Observed GOES-16 CLAVR-x Cloud Top Heights\nValid: {file_time_str}', fontsize=title_fontsize)
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax1.add_feature(cfeature.STATES, linewidth=0.5)
    gridlines = ax1.gridlines(draw_labels=True, crs=ccrs.PlateCarree())
    gridlines.right_labels = False
    gridlines.top_labels = False
    gridlines.xlabel_style = {'fontsize': label_fontsize}
    gridlines.ylabel_style = {'fontsize': label_fontsize}
    gridlines.xlabels_bottom = True
    gridlines.ylabels_left = True
    gridlines.xlabels_top = False
    gridlines.ylabels_right = False
    gridlines.xlabel_style.update({'fontsize': tick_fontsize})
    gridlines.ylabel_style.update({'fontsize': tick_fontsize})

    # Plot Nowcast CTH
    cth_data = np.copy(CBH_CTH_nowcast_array[idx, :, :, 1])
    cth_data = np.ma.masked_invalid(cth_data)  # Mask NaN values for plotting
    im2 = ax2.imshow(
        cth_data, cmap='viridis_r', 
        origin='upper', vmin=0, vmax=15000,
        extent=extent,transform=geo_proj)

    ax2.set_title(f'I-NOW ({idx*time_interval:02} min) Cloud Top Heights [m]\nValid: {file_time_str}', fontsize=title_fontsize)
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax2.add_feature(cfeature.STATES, linewidth=0.5)
    gridlines = ax2.gridlines(draw_labels=True, crs=ccrs.PlateCarree())
    gridlines.right_labels = False
    gridlines.top_labels = False
    gridlines.xlabel_style = {'fontsize': label_fontsize}
    gridlines.ylabel_style = {'fontsize': label_fontsize}
    gridlines.xlabels_bottom = True
    gridlines.ylabels_left = True
    gridlines.xlabels_top = False
    gridlines.ylabels_right = False
    gridlines.xlabel_style.update({'fontsize': tick_fontsize})
    gridlines.ylabel_style.update({'fontsize': tick_fontsize})

    # Add a single shared colorbar below the plots
    cbar_ax = fig.add_subplot(spec[1, :])  # Single subplot spanning both columns
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Cloud Top Heights [m]', fontsize=colorbar_fontsize)
    cbar.ax.tick_params(labelsize=colorbar_fontsize)

    # Save the figure
    os.makedirs(f'{base_dir}/CTH_Plots/', exist_ok=True)  # Ensure the directory exists
    plt.savefig(f'{base_dir}/CTH_Plots/Nowcast{idx:02}.png', bbox_inches='tight')
    plt.close()

    # Update file_time for the next iteration
    file_time = file_time + datetime.timedelta(minutes=time_interval)
    file_time_str = datetime.datetime.strftime(file_time,'%m/%d/%Y %H:%M UTC')

#Make Loop
gif_loop_maker(f'{base_dir}/CTH_Plots/', frame_duration = 300, loop_count = 0) #Frame duration in milliseconds, loop_count = 0 is infinite loop

#%%
#Plotting CBH Nowcast Results and Truth Data

# Setting colormap for CBH Plots (need to set 'none' for masked values)
cmap = plt.colormaps['viridis_r'].copy()
cmap.set_bad(color='none') 

# Plotting
file_time = initial_time
file_time_str = datetime.datetime.strftime(file_time,'%m/%d/%Y %H:%M UTC')
for idx in range(CBH_CTH_nowcast_array.shape[0]):
    print(f'Working on image {idx:02}.')

    # Define figure and GridSpec layout
    fig = plt.figure(figsize=(18, 10))
    spec = gridspec.GridSpec(ncols=2, nrows=2, height_ratios=[1, 0.05], hspace=0.1, wspace=0.05)

    # Create subplots for the images
    ax1 = fig.add_subplot(spec[0, 0], projection=geo_proj)
    ax2 = fig.add_subplot(spec[0, 1], projection=geo_proj)

    # Define extent based on data's spatial bounds
    extent = [
        sat_parameters['x_meters_min'],
        sat_parameters['x_meters_max'],
        sat_parameters['y_meters_min'],
        sat_parameters['y_meters_max']
    ]

    # Plot True CBH on left
    cbh_data_truth = np.copy(CLAVRx_Truth_array[idx, :, :, 0])
    cbh_data_truth = ma.masked_invalid(cbh_data_truth)  # Mask NaN values for plotting
    im1 = ax1.imshow(
        cbh_data_truth, cmap='viridis_r', 
        origin='upper', vmin=0, vmax=15000,
        extent=extent,transform=geo_proj)
    
    ax1.set_title(f'Observed GOES-16 CLAVR-x Cloud Base Heights [m]\nValid: {file_time_str}', fontsize=title_fontsize)
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax1.add_feature(cfeature.STATES, linewidth=0.5)
    gridlines = ax1.gridlines(draw_labels=True, crs=ccrs.PlateCarree())
    gridlines.right_labels = False
    gridlines.top_labels = False
    gridlines.xlabel_style = {'fontsize': label_fontsize}
    gridlines.ylabel_style = {'fontsize': label_fontsize}
    gridlines.xlabels_bottom = True
    gridlines.ylabels_left = True
    gridlines.xlabels_top = False
    gridlines.ylabels_right = False
    gridlines.xlabel_style.update({'fontsize': tick_fontsize})
    gridlines.ylabel_style.update({'fontsize': tick_fontsize})

    # Plot Nowcast CTH on right
    cbh_data = np.copy(CBH_CTH_nowcast_array[idx, :, :, 0])
    cbh_data = ma.masked_invalid(cbh_data)  # Mask NaN values for plotting
    im2 = ax2.imshow(
        cbh_data, cmap='viridis_r', origin='upper',
        vmin=0, vmax=15000,
        extent=extent,
        transform=geo_proj
    )

    ax2.set_title(f'I-NOW ({idx*time_interval:02} min) Cloud Base Heights [m]\nValid: {file_time_str}', fontsize=title_fontsize)
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax2.add_feature(cfeature.STATES, linewidth=0.5)
    gridlines = ax2.gridlines(draw_labels=True, crs=ccrs.PlateCarree())
    gridlines.right_labels = False
    gridlines.top_labels = False
    gridlines.xlabel_style = {'fontsize': label_fontsize}
    gridlines.ylabel_style = {'fontsize': label_fontsize}
    gridlines.xlabels_bottom = True
    gridlines.ylabels_left = True
    gridlines.xlabels_top = False
    gridlines.ylabels_right = False
    gridlines.xlabel_style.update({'fontsize': tick_fontsize})
    gridlines.ylabel_style.update({'fontsize': tick_fontsize})

    # Add a single shared colorbar below the plots
    cbar_ax = fig.add_subplot(spec[1, :])  # Single subplot spanning both columns
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Cloud Base Heights [m]', fontsize=colorbar_fontsize)
    cbar.ax.tick_params(labelsize=colorbar_fontsize)

    # Save the figure
    os.makedirs(f'{base_dir}/CBH_Plots/', exist_ok=True)  # Ensure the directory exists
    plt.savefig(f'{base_dir}/CBH_Plots/Nowcast{idx:02}.png', bbox_inches='tight')
    plt.close()

    # Update file_time for the next iteration
    file_time = file_time + datetime.timedelta(minutes=time_interval)
    file_time_str = datetime.datetime.strftime(file_time,'%m/%d/%Y %H:%M UTC')

#Make Loop
gif_loop_maker(f'{base_dir}/CBH_Plots/', frame_duration = 300, loop_count = 0) #Frame duration in milliseconds, loop_count = 0 is infinite loop

# %%
