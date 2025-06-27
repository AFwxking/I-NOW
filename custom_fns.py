#%%
#Script to hold all necessary python functions

#Necessary imports
import numpy as np
import datetime 
import glob
import xarray as xr
import s3fs
import os
import requests

#Imports for plots
from PIL import Image

#Function to get CLAVR-x file path based on satellite name, scan type, date, time, and save directory
def get_clavrx_file_path(sat_name, scan, date, time, save_dir):
    '''Function that takes in satellite name, scan type, date, time, and save directory and returns the path to the CLAVR-x file
    Inputs:
    sat_name: string of satellite name (e.g. 'G16', 'G17', 'G18', 'goes16', 'goes17', 'goes18')
    scan: string of scan type (e.g. 'RadC', 'RadF')
    date: string of date in the form 'mm/dd/yyyy'
    time: string of time in the form 'hh:mm'
    save_dir: string of save directory where the CLAVR-x file is saved
    Outputs:
    clavrx_filenames: list of strings of CLAVR-x file paths
    '''

    try: 
        datetime_str = f'{date} {time}'
        datetime_object = datetime.datetime.strptime(datetime_str, '%m/%d/%Y %H:%M')
    except Exception as e:
        print(f'Make sure date is in the form mm/dd/yyyy and time is in the form hh:mm: {e}')
        return

    #Check for satellite
    if ('G16' in sat_name) or ('goes16' in sat_name):
        long_sat_str = 'goes16'
        short_sat_str = 'G16'
    elif ('G17' in sat_name) or ('goes17' in sat_name):
        long_sat_str = 'goes17'
        short_sat_str = 'G17'
    elif ('G18' in sat_name) or ('goes18' in sat_name):
        long_sat_str = 'goes18'
        short_sat_str = 'G18'

    #Check what scan
    if 'RadC' in scan:
        scan_str = 'RadC'
    elif 'RadF' in scan:
        scan_str = 'RadF'

    #Get julian day needed for later
    ti = datetime_object.timetuple()
    julian_day = ti.tm_yday

    #Predefine filename list
    clavrx_filenames = []

    #Define destination path    
    destination_path = f'{save_dir}clavrx_files/{scan_str}/{datetime_object.year}/{datetime_object.year}_{datetime_object.month:02}_{datetime_object.day:02}/'

    #Check for file before moving
    file_test = sorted(glob.glob(f'{destination_path}clavrx_OR_ABI-L1b-{scan_str}*{short_sat_str}_s{datetime_object.year}{julian_day:03}{datetime_object.hour:02}{datetime_object.minute:02}*.level2.hdf'))
    file_test2 = sorted(glob.glob(f'{destination_path}clavrx_{long_sat_str}_{datetime_object.year}_{julian_day:03}_{datetime_object.hour:02}{datetime_object.minute:02}*.level2.hdf'))
    if len(file_test)!= 0:
        print('File already exists locally.')
        #Add filename to list
        clavrx_filenames.append(sorted(glob.glob(f'{destination_path}clavrx_OR_ABI-L1b-{scan_str}*{short_sat_str}_s{datetime_object.year}{julian_day:03}{datetime_object.hour:02}{datetime_object.minute:02}*.level2.hdf')))
        
    elif len(file_test2) != 0:
        print('File already exists locally.')
        #Add filename to list
        clavrx_filenames.append(sorted(glob.glob(f'{destination_path}clavrx_{long_sat_str}_{datetime_object.year}_{julian_day:03}_{datetime_object.hour:02}{datetime_object.minute:02}*.level2.hdf')))
        
    else:
        print("File doesn't already exist in local directory.")

        return

    return clavrx_filenames

def clavrx_truth(file_dir):
    '''Function that takes in clavrx file directory, opens file, and returns cloud top height and cloud base height values
    Input:
    file_dir: string of clavrx file directory
    Output:
    CTH_values
    CBH_values:
    '''

    #Reading Clavrx Hgt Data
    try:
        Clavrx_xarray = xr.open_dataset(file_dir, engine='netcdf4')
    except Exception as e:
        print(f'Error opening the file: {e}')
        return
    
    try:
        #Get Cloud Top Height and Cloud Base Heights from clavrx files
        CBH_values = Clavrx_xarray['cld_height_base_acha'].data
        CTH_values = Clavrx_xarray['cld_height_acha'].data
        CLD_MSK_values = Clavrx_xarray['cloud_mask'].data #Flag values: [0 1 2 3] Flag Meanings [clear, probably_clear, probably_cloud, cloudy]
        CLD_MSK_values[CLD_MSK_values == -128] = np.nan
        CTH_values[CLD_MSK_values < 2] = np.nan
        CBH_values[CLD_MSK_values < 2] = np.nan
    except Exception as e:
        print(f'Error with a variable name...trying different variable names')
        
        #Get Cloud Top Height and Cloud Base Heights from clavrx files...with different variable names if required
        try:
            CBH_values = Clavrx_xarray['cld_height_base'].data
            CTH_values = Clavrx_xarray['cld_height_acha'].data
            CLD_MSK_values = Clavrx_xarray['cloud_mask'].data #Flag values: [0 1 2 3] Flag Meanings [clear, probably_clear, probably_cloud, cloudy]
            CLD_MSK_values[CLD_MSK_values == -128] = np.nan
            CTH_values[CLD_MSK_values < 2] = np.nan
            CBH_values[CLD_MSK_values < 2] = np.nan
        except Exception as e:
            return
            
    return CTH_values, CBH_values


def generate_goes_urls(sat_name, scan, date, time, channel):
    '''Function to generate URLs for GOES data from AWS based on satellite name, scan type, date, time, and channel
    Inputs:
    sat_name: string of satellite name (e.g. 'G16', 'G17', 'G18')
    scan: string of scan type (e.g. 'RadC', 'RadF')
    date: string of date in the form 'mm/dd/yyyy'
    time: string of time in the form 'hh:mm'
    channel: string of channel (e.g. '13', '14', etc.)
    Outputs:
    urls: list of strings of URLs for matching files
    '''

    # Validate and parse date and time
    datetime_str = f'{date} {time}'
    try:
        datetime_object = datetime.datetime.strptime(datetime_str, '%m/%d/%Y %H:%M')
    except ValueError as e:
        print(f'Error parsing date/time: {e}')
        return None
    
    # Set satellite and product path
    sat_dict = {'G16': 'noaa-goes16', 'G17': 'noaa-goes17', 'G18': 'noaa-goes18'}
    if sat_name not in sat_dict:
        print('Invalid satellite name. Choose from G16, G17, or G18.')
        return None
    sat_path = sat_dict[sat_name]

    # Define product and scan type
    product = f'ABI-L1b-Rad{scan[-1]}'

    # Julian Day calculation
    julian_day = datetime_object.timetuple().tm_yday
    
    # S3 filesystem with anonymous access
    fs = s3fs.S3FileSystem(anon=True)

    # Construct the S3 path
    year = datetime_object.year
    hour = f"{datetime_object.hour:02}"
    prefix = f"{product}/{year}/{julian_day:03}/{hour}/"

    # List files in the specified prefix
    try:
        file_list = fs.ls(f"s3://{sat_path}/{prefix}")
    except Exception as e:
        print(f"Error listing files: {e}")
        return False

    # Generate URLs for matching files, filtering by exact time and channel
    urls = []
    for file in file_list:
        if (f"{sat_name}_s{year}{julian_day:03}{hour}{datetime_object.minute:02}" in file and 
            f"C{channel}" in file):
            # Create the initial URL with the bucket name
            url = f"https://{sat_path}.s3.amazonaws.com/{file.replace(f's3://{sat_path}/', '')}"
            
            # Remove redundant bucket name part if it appears in the path
            cleaned_url = url.replace(f'https://{sat_path}.s3.amazonaws.com/{sat_path}/', f'https://{sat_path}.s3.amazonaws.com/')
            
            urls.append(cleaned_url)
    if len(urls)== 0:
        return False

    return urls

def get_sat_data_aws(sat_name, scan, date, time, save_dir, channel):
    '''Function to download GOES data from AWS based on satellite name, scan type, date, time, and save directory
    Inputs:
    sat_name: string of satellite name (e.g. 'G16', 'G17', 'G18')
    scan: string of scan type (e.g. 'RadC', 'RadF')
    date: string of date in the form 'mm/dd/yyyy'
    time: string of time in the form 'hh:mm'
    save_dir: string of save directory where the GOES data will be saved
    channel: string of channel (e.g. '13', '14', etc.)
    Outputs:
    filenames: list of strings of downloaded file paths
    '''
    
    try:
        datetime_str = f'{date} {time}'
        datetime_object = datetime.datetime.strptime(datetime_str, '%m/%d/%Y %H:%M')
    except Exception as e:
        print(f"Make sure date is in the form mm/dd/yyyy and time is in the form hh:mm: {e}")
        return
    
    # Julian Day calculation
    julian_day = datetime_object.timetuple().tm_yday

    #Check for satellite in file_dir
    if ('G16' in sat_name) or ('goes16' in sat_name):
        long_sat_str = 'goes16'
    elif ('G17' in sat_name) or ('goes17' in sat_name):
        long_sat_str = 'goes17'
    elif ('G18' in sat_name) or ('goes18' in sat_name):
        long_sat_str = 'goes18'
    else:
        print("Invalid satellite name")
        return

    # Scan type selection
    if 'RadC' in scan:
        scan_str = 'RadC'
    elif 'RadF' in scan:
        scan_str = 'RadF'
    else:
        print("Invalid scan type")
        return

    # Get URLs for matching files
    urls = generate_goes_urls(sat_name, scan, date, time, channel)
    if not urls:
        print("No matching URLs found.")
        return False

    # Download the files
    filenames = []
    destination_path = f'{save_dir}{long_sat_str}/{scan_str}/{datetime_object.year}/{datetime_object.year}_{datetime_object.month:02}_{datetime_object.day:02}_{julian_day:03}/'
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    local_file_path = os.path.join(destination_path, os.path.basename(urls[0]))
    if os.path.exists(local_file_path):
        print(f'{local_file_path} already exists...')
        filenames.append(local_file_path)
    else:
        try:
            # Download using HTTP
            response = requests.get(urls[0])
            response.raise_for_status()  # Raise an error for bad responses
            with open(local_file_path, 'wb') as file:
                file.write(response.content)
            filenames.append(local_file_path)
            print(f"Downloaded {local_file_path}")
        except Exception as e:
            print(f"Failed to download {urls[0]}: {e}")

    return filenames

def get_sat_parameters(file_dir):

    '''Function that takes in a GOES file directory, opens file, and returns satellite parameters

    Input:
    file_dir: string of GOES file directory
    Output:
    sat_parameters: dictionary with satellite parameters
    '''

    ds = xr.open_dataset(file_dir, engine='netcdf4')
    x = ds['x'].data
    y = ds['y'].data
    sat_height = ds['goes_imager_projection'].perspective_point_height  # in m
    sat_lon = ds['goes_imager_projection'].longitude_of_projection_origin  # in degrees
    x_meters = x * sat_height 
    y_meters = y * sat_height
    x_meters_min = np.min(x_meters)
    x_meters_max = np.max(x_meters)
    y_meters_min = np.min(y_meters)
    y_meters_max = np.max(y_meters)

    sat_parameters = {
        'sat_height': sat_height,  # in km
        'sat_lon': sat_lon,  # in degrees
        'x_meters_min': x_meters_min,  # in meters
        'x_meters_max': x_meters_max,  # in meters
        'y_meters_min': y_meters_min,  # in meters
        'y_meters_max': y_meters_max,  # in meters
    }

    return sat_parameters

def gif_loop_maker(file_dir, frame_duration = 200, loop_count = 0):
    '''Function that takes in a directory of png files and creates a gif loop
    Input:
    file_dir: string of directory with png files
    Output:
    gif_filename: string of gif filename
    '''

    #Make loop
    png_flist = sorted(glob.glob(f'{file_dir}/*.png'))

    # Open the first image to get its size
    first_image = Image.open(png_flist[0])
    width, height = first_image.size

    # Create a list to store the frames
    frames = []

    # Open each image, convert it to RGBA mode (if not already), and append to frames
    for png_filename in png_flist:
        img = Image.open(png_filename)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        frames.append(img)

    # Save the animated GIF
    gif_filename = f'{file_dir}/loop.gif'
    frames[0].save(
        gif_filename,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration, # Frame duration in milliseconds, adjust this as needed
        loop=loop_count  # 0 means infinite loop, adjust as needed
    )

# %%
