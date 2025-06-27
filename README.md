Purpose: Script to provide example code for I-NOW (Improved Nowcasting via Optical flow Warping) presented in "Nowcasting 3D Cloud Fields Using Forward Warping Optical Flow" by Matthew King, Jason Apke, Steven Miller, Katherine Haynes, Yoo-Jeong Noh, and John Haynes, 2025: J. Atmos Ocea. Tech.
DOI: TBD

Requires: 
    - Python 3.12.2 (version used for development, other versions may work)
    - Data files which include CLAVRx_data and GOES Full Disk Band 13 data (code to download GOES data from AWS is included)
    - Python scripts contained in OCTOPY folder (contains scripts to run python version of OCTANE optical flow code and the I-NOW nowcast method which requires temporal interpolation scripts)
    - Code requires GPU with support for CUDA (developed on NVIDIA RTX A6000)
    - Data files within the DOI in data availability statement of manuscript (CLAVRx CTH/CBH data saved in clavrx_data folder on GitHub Repo)

Dependencies: netcdf4, numba, cudatookit, s3fs, matplotlib, cartopy, xarray, metpy
    - netcdf4, numba, and cudatoolkit required to run nowcasts
    - s3fs required to download GOES data from AWS
    - xarray, matplotlib, cartopy, and metpy required in code to plot results

Recommend using a virtual environment to run this code, such as conda.
Development required install using the following commands:
    create an anaconda environment:
        $ conda create -n INOW_env python=3.12.2
        $ conda activate INOW_env
    Install Netcdf4 first, this one does not like to play nice with the others:
        $ conda install -c conda-forge netcdf4
    Install numba
        $ conda install -c numba numba
    Install the cudatoolkit, use the anaconda distribution, the conda-forge one has a driver issue
        $ conda install -c anaconda cudatoolkit
    Now matplotlib, use the pip-install for this
        $ python -m pip install -U matplotlib
    Now cartopy
        $ conda install -c conda-forge cartopy
    Now xarray
        $ conda install -c conda-forge xarray
    Now metpy
        $ conda install -c conda-forge metpy
    Now s3fs
        $ conda install -c conda-forge s3fs

environment.yml file provided in directory for easy installation of all dependencies
    $ conda env create -f environment.yml

Example run commnand (unix):
    $ python Nowcast_Example.py

Code Author: Matthew P. King, Lt Col, USAF
Date: 27 June 2025 
Version: 1.0

Explanation of code: 
Nowcast_Example.py provides an example of how to run the I-NOW nowcasting method on GOES-16 Band 13 data. Code downloads GOES-16 Band 13 data from AWS, runs a python version of OCTANE optical flow code to calculate the optical flow using the values contained in the JTEC article. Rather than linear extrapolation, the code uses a warping method formed from computer vision temporal interpolation methods to account for occlusions and time related changes to the optical flow field in the forward time direction rather than a time that is intermediate of two images. The code provides nowcasts of GOES-16 10.3 micron brightness temperature,CLAVR-x derived cloud top heights (CTH), and CLAVR-x derived cloud base heights (CBH). Observed brightness temperature data and CLAVR-x data are plotted alongside the nowcast results in 10-minute intervals. The code also creates animated gifs of the nowcaststo easier visualize the results through time. The code is designed to be run on a GPU with CUDA support, and the user can define the save directory for the results to be saved and data that is downloaded. CLAVR-x data is not downloadable from the internet,so the GitHub repository contains a folder with CLAVR-x data for the example date and time used in the code that has been reduced to contain only the required CTH and CBH data instead of full CLAVR-x retrieval data.  
