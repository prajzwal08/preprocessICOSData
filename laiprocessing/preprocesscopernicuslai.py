import os
import xarray as xr
import numpy as np
import pandas as pd
from smoothing import smoothing_LAI
from utils import resampleLAI_to_fluxtower_resolution


def get_copernicusLAI_for_station(longitude,latitude,time_interval = "30min"):
    file_path_copernicus_lai = "/home/khanalp/data/copernicus_lai/"
    nc_files = [os.path.join(root, file) for root, dirs, files in os.walk(file_path_copernicus_lai) for file in files if file.endswith('.nc')]
    copernicus_lai_global = xr.open_mfdataset(nc_files)
    # Find the central grid cell by selecting the nearest latitude and longitude
    #Since data until 2020 is not completed only, until 2019.
    grid_containing_coordinate = copernicus_lai_global.sel(lat=latitude, lon=longitude, method='nearest')
    # Find the index of grid_containing_coordinate in copernicus_lai_global
    idx_lon = np.argmin(np.abs(copernicus_lai_global.lon.values - grid_containing_coordinate.lon.values))
    idx_lat = np.argmin(np.abs(copernicus_lai_global.lat.values - grid_containing_coordinate.lat.values))
    #selecting neighboring nine surrounding grid cells
    nine_neighboring_grids = copernicus_lai_global.sel(lon = slice(copernicus_lai_global.lon.values[idx_lon - 1],copernicus_lai_global.lon.values[idx_lon + 1]),
                                       lat = slice(copernicus_lai_global.lat.values[idx_lat - 1],copernicus_lai_global.lat.values[idx_lat + 1])).sel(time=slice(None,pd.Timestamp('2019-12-31')))
    #getting spatial average
    spatial_average_lai = nine_neighboring_grids.mean(dim=['lat', 'lon'], skipna = True)
    
    #Getting array in the shape where each row represents a year so it could be passed into smoothing function
    unsmooth_lai = spatial_average_lai.LAI.values.reshape(-1,36)
    
    #Smoothing (getting mean climatology, smoothing anomaly (+/- 6 tsteps) and adding the mean climatology back)
    smooth_lai = smoothing_LAI(unsmooth_lai)
    
    df_lai_original = pd.DataFrame({'Date':spatial_average_lai.time, 'LAI' :smooth_lai})
    
    # Convert 'Date' column to datetime
    df_lai_original['Date'] = pd.to_datetime(df_lai_original['Date'])

    # Set 'Date' column as index
    df_lai_original.set_index('Date', inplace=True)
    
    df_lai_resampled = resampleLAI_to_fluxtower_resolution(data_frame=df_lai_original,
                                                                        resampling_interval = "30min")
    return df_lai_resampled
    





