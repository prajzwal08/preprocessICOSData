import glob 
import pandas as pd
import numpy as np
from spatial_weighing import get_spatial_weighted_LAI
from interpolation import interpolate_NA_LAI
from  data_availability import check_data_availability_LAI
from smoothing import smoothing_LAI
from utils import resampleLAI_to_fluxtower_resolution
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


def get_modisLAI_for_station(station_name,time_interval = "30min"):
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
    modis_path = "/home/khanalp/data/MODIS_Raw/"
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

    df_lai_resampled = resampleLAI_to_fluxtower_resolution(data_frame=df_lai_original,
                                                                           resampling_interval = time_interval)
    return df_lai_resampled

