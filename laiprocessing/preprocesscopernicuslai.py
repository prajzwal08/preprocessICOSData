# This script retrieves and processes Copernicus LAI (Leaf Area Index) data for a given station using its geographic coordinates (longitude and latitude).
# The script reads LAI data from multiple NetCDF files, selects the nearest grid cell, calculates the spatial average of the surrounding nine cells,
# fills missing values, applies smoothing, and resamples the data to a desired time resolution.

import os
import xarray as xr
import numpy as np
import pandas as pd
from utils import smoothing_LAI
from utils import interpolate_NA_LAI
from utils import resampleLAI_to_fluxtower_resolution

def get_copernicusLAI_for_station(longitude, latitude, time_interval="30min"):
    # Define the file path for Copernicus LAI data (change to v0 if needed)
    file_path_copernicus_lai = "/home/khanalp/data/copernicus_lai/v0"
    
    # Retrieve all NetCDF (.nc) files in the specified directory and its subdirectories
    nc_files = [os.path.join(root, file) for root, dirs, files in os.walk(file_path_copernicus_lai) for file in files if file.endswith('.nc')]
    
    # Open all NetCDF files as a single xarray dataset
    copernicus_lai_global = xr.open_mfdataset(nc_files)

    # Select the end date for the data, up to 2019 since data after 2020 is incomplete
    end_date = pd.Timestamp('2019-12-31')

    # Find the grid cell closest to the specified latitude and longitude
    grid_containing_coordinate = copernicus_lai_global.sel(lat=latitude, lon=longitude, method='nearest')
    
    # Determine the index of the closest grid cell
    idx_lon = np.argmin(np.abs(copernicus_lai_global.lon.values - grid_containing_coordinate.lon.values))
    idx_lat = np.argmin(np.abs(copernicus_lai_global.lat.values - grid_containing_coordinate.lat.values))
    
    # Select the nine surrounding grid cells (3x3) around the closest grid cell
    nine_neighboring_grids = copernicus_lai_global.sel(
        lon=slice(copernicus_lai_global.lon.values[idx_lon - 1], copernicus_lai_global.lon.values[idx_lon + 1]),
        lat=slice(copernicus_lai_global.lat.values[idx_lat - 1], copernicus_lai_global.lat.values[idx_lat + 1])
    ).sel(time=slice(None, end_date))

    # Calculate the spatial average LAI across the nine grid cells
    spatial_average_lai = nine_neighboring_grids.LAI.mean(dim=['lat', 'lon'], skipna=True).values  # This contains data gaps for v0 data.
    
    # Interpolate to fill missing values in the spatially averaged LAI
    filled_lai = interpolate_NA_LAI(spatial_average_lai)
    
    # Reshape the LAI array to have each row represent a year (36 time steps per year)
    unsmooth_lai = spatial_average_lai.reshape(-1, 36)
    
    # Apply smoothing to the LAI data (mean climatology, smoothing anomaly, and adding back the mean climatology)
    smooth_lai = smoothing_LAI(unsmooth_lai)
    
    # Create a DataFrame with the original time series of smoothed LAI data
    df_lai_original = pd.DataFrame({'Date': nine_neighboring_grids.time, 'LAI': smooth_lai})
    
    # Convert 'Date' column to datetime format
    df_lai_original['Date'] = pd.to_datetime(df_lai_original['Date'])

    # Set 'Date' column as the DataFrame index
    df_lai_original.set_index('Date', inplace=True)
    
    # Resample the LAI data to the desired time resolution (e.g., "30min")
    df_lai_resampled = resampleLAI_to_fluxtower_resolution(data_frame=df_lai_original, resampling_interval=time_interval)
    
    # Return the resampled LAI DataFrame
    return df_lai_resampled
