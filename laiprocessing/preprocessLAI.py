import glob 
import pandas as pd
import numpy as np
from laiprocessing.spatial_weighing import get_spatial_weighted_LAI
from laiprocessing.interpolation import interpolate_NA_LAI
from  laiprocessing.data_availability import check_data_availability_LAI
from laiprocessing.smoothing import smoothing_LAI
import matplotlib.pyplot as plt


def extract_pixel_data(data_frame, no_tsteps, pixel_no):
    """
    Extracts LAI, standard deviation (SD), and quality control (QC) data for a specific pixel.

    Parameters:
        lai (pandas.DataFrame): DataFrame containing LAI data.
        sd (pandas.DataFrame): DataFrame containing SD data.
        qc (pandas.DataFrame): DataFrame containing QC data.
        no_tsteps (int): Number of timesteps.
        pixel_no (int): Pixel number.

    Returns:
        tuple: A tuple containing lai_pixel, sd_pixel, and qc_pixel arrays.
    """
    # Check if the 'scale' column contains numeric values (because for qc its not available.)
    if data_frame['scale'].dtype.kind in 'iufc':  # 'iufc' represents integer, unsigned integer, float, and complex types
        # If 'scale' column contains numeric values, apply scaling
        pixel_array = np.full((no_tsteps, len(pixel_no)), np.nan)
        for idx, p in enumerate(pixel_no):
            pixel_array[:, idx] = data_frame.loc[data_frame['pixel'] == p, 'value'].values[:no_tsteps] * data_frame.loc[data_frame['pixel'] == p, 'scale'].values[0]
    else:
        # If 'scale' column doesn't contain numeric values, just extract 'value' column
        pixel_array = np.full((no_tsteps, len(pixel_no)), np.nan)
        for idx, p in enumerate(pixel_no):
            pixel_array[:, idx] = data_frame.loc[data_frame['pixel'] == p, 'value'].values[:no_tsteps]

    return pixel_array

import pandas as pd

def resampleLAI_to_fluxtower_resolution(data_frame, start_date, end_date, resampling_interval='30min'):
    """
    Resamples LAI data to the flux tower resolution.

    Parameters:
        data_frame (pandas.DataFrame): DataFrame containing LAI data with a 'Date' column.
        start_date (str): Start date for the resampling.
        end_date (str): End date for the resampling.
        resampling_interval (str, optional): Resampling interval. Defaults to '30 min'.

    Returns:
        pandas.DataFrame: Resampled and filtered LAI data.
    """
    
    # Resample to specified interval and forward fill missing values
    df_filled = data_frame.resample(resampling_interval).ffill()

    # Get the year of the last date in the DataFrame
    last_date_year = df_filled.index[-1].year
    
    # Reindex to extend the index until the end of the last year and forward fill
    end_date_extend = pd.to_datetime(f'{last_date_year}-12-31 23:30:00')
    df_filled = df_filled.reindex(pd.date_range(start=df_filled.index.min(), end=end_date_extend, freq=resampling_interval)).ffill()
    
    # Filter DataFrame based on start_date and end_date
    filtered_df = df_filled[(df_filled.index >= start_date) & (df_filled.index <= end_date)]

    return filtered_df


def get_LAI_for_station(modis_path,station_name,start_date,end_date, time_interval = "30min"):
    """
    Retrieves Leaf Area Index (LAI) data from MODIS files for a specific station and time range. 
    The function performs data preprocessing steps including spatial weighting, interpolating missing values, 
    checking data availability, smoothing LAI, and resampling to match the resolution of the flux tower data.
    
    Parameters:
        modis_path (str): Path to the directory containing MODIS files.
        station_name (str): Name of the station.
        start_date (str): Start date of the desired time range (format: 'YYYY-MM-DD').
        end_date (str): End date of the desired time range (format: 'YYYY-MM-DD').
    
    Returns:
        numpy.ndarray: Array containing the smoothed LAI values.
    """
    
    lai_file = glob.glob(f"{modis_path}/{station_name}_MCD15A2H_Lai_500m_*")
    qc_file = glob.glob(f"{modis_path}/{station_name}_MCD15A2H_FparLai_QC*")
    sd_file = glob.glob(f"{modis_path}/{station_name}_MCD15A2H_LaiStdDev_500m_*")
    
    df_lai = pd.read_csv(lai_file[0])
    df_sd = pd.read_csv(sd_file[0])
    df_qc = pd.read_csv(qc_file[0])


    # Get the number of timesteps
    no_tsteps = min(len(df_lai), len(df_sd), len(df_qc)) // max(df_lai['pixel'])

    # Extracting pixels in the centre and immediately around it
    pixel_no = [7, 8, 9, 12, 13, 14, 17, 18, 19]
    
    # Save time stamps
    lai_time = pd.to_datetime(df_lai.loc[df_lai['pixel'] == pixel_no[0], 'calendar_date'])

    
    # Extract pixel data:
    lai_pixel = extract_pixel_data(df_lai,no_tsteps=no_tsteps,pixel_no=pixel_no)
    sd_pixel = extract_pixel_data(df_sd,no_tsteps=no_tsteps,pixel_no=pixel_no)
    qc_pixel = extract_pixel_data(df_qc,no_tsteps=no_tsteps,pixel_no=pixel_no)

    #print("Spatial Weighing started")
    weighted_lai_values = get_spatial_weighted_LAI(lai_pixel,sd_pixel,qc_pixel)
        
    #print("Interpolating NAs")
    filled_lai = interpolate_NA_LAI(weighted_lai_values)
    
    #print("checking data availability")
    gap_free_lai, selected_dates = check_data_availability_LAI(filled_lai,lai_time, start_year=2003,end_year=2023)

    #print("Smoothing LAI")
    smooth_lai = smoothing_LAI(gap_free_lai)
    
    df_lai_original = pd.DataFrame({'Date':selected_dates, 'LAI' :smooth_lai})
    # Convert 'Date' column to datetime
    df_lai_original['Date'] = pd.to_datetime(df_lai_original['Date'])

    # Set 'Date' column as index
    df_lai_original.set_index('Date', inplace=True)
    

    lai_resampled_to_flux_resolution = resampleLAI_to_fluxtower_resolution(data_frame=df_lai_original,
                                                                           start_date=start_date,
                                                                           end_date=end_date,
                                                                           resampling_interval = time_interval)
    return lai_resampled_to_flux_resolution['LAI'].values


