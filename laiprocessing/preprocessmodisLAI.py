# This script processes Leaf Area Index (LAI) data from MODIS files for a specific station.
# It performs various steps including extracting pixel-level data, applying spatial weighting,
# filling missing values, checking data availability, smoothing the data, and resampling it
# to match the resolution of the flux tower data.

import os
import pandas as pd
import numpy as np
from utils import interpolate_NA_LAI
from utils import smoothing_LAI
from utils import resampleLAI_to_fluxtower_resolution
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def extract_pixel_data(data_frame, no_tsteps, pixel_no):
    """
    Extracts LAI, standard deviation (SD), and quality control (QC) data for specific pixels.

    Parameters:
        data_frame (pandas.DataFrame): DataFrame containing LAI, SD, or QC data.
        no_tsteps (int): Number of timesteps.
        pixel_no (list): List of pixel numbers.

    Returns:
        numpy.ndarray: Array of extracted pixel values, scaled if applicable.
    """
    # Initialize an array to store pixel data
    pixel_array = np.full((no_tsteps, len(pixel_no)), np.nan)

    # Check if 'scale' column contains numeric values for scaling (QC data does not have scaling)
    if data_frame['scale'].dtype.kind in 'iufc':  # 'iufc' represents integer, unsigned integer, float, and complex types
        # Apply scaling if 'scale' column contains numeric values
        for idx, p in enumerate(pixel_no):
            pixel_array[:, idx] = data_frame.loc[data_frame['pixel'] == p, 'value'].values[:no_tsteps] * data_frame.loc[data_frame['pixel'] == p, 'scale'].values[0]
    else:
        # Extract values without scaling
        for idx, p in enumerate(pixel_no):
            pixel_array[:, idx] = data_frame.loc[data_frame['pixel'] == p, 'value'].values[:no_tsteps]

    return pixel_array

def get_spatial_weighted_LAI(lai_pixel, sd_pixel, qc_pixel):
    """
    Calculates the spatially weighted Leaf Area Index (LAI) using LAI values, standard deviations (SD), and quality control (QC) flags.

    Parameters:
        lai_pixel (numpy.ndarray): Array of LAI values for each pixel.
        sd_pixel (numpy.ndarray): Array of standard deviation values for each pixel.
        qc_pixel (numpy.ndarray): Array of quality control flags for each pixel.

    Returns:
        numpy.ndarray: Array containing the spatially weighted LAI values.
    """
    # Create copies to preserve original data
    lai_copy = np.copy(lai_pixel)
    sd_copy = np.copy(sd_pixel)

    # Define poor data quality flags
    qc_flags = [0, 2, 24, 26, 32, 34, 56, 58]
    
    # Mask out poor quality data based on QC flags
    mask = np.isin(qc_pixel, qc_flags)
    lai_copy[~mask] = np.nan
    sd_copy[~mask] = np.nan
    
    # Mask out areas where SD is low (likely cloud effects)
    sd_copy[sd_copy < 0.1] = np.nan
    lai_copy[np.isnan(sd_copy)] = np.nan
    
    # Set values above a certain threshold to NaN
    lai_copy[lai_copy > 10] = np.nan
    
    # Calculate spatial weights, ignoring NaN values
    weights = (1 / sd_copy**2) / np.nansum(1 / sd_copy**2, axis=1, keepdims=True)
    
    # Calculate weighted LAI values
    weighted_lai_values = lai_copy * weights
    
    # Compute the weighted mean for each row, ignoring NaN values
    weighted_lai = np.nansum(weighted_lai_values, axis=1)
    
    return weighted_lai

def check_data_availability_LAI(filled_lai, lai_time, start_year, end_year):
    """
    Checks the availability of MODIS LAI data and fills missing values if necessary.
    Ensures that the provided time range matches the expected range for MODIS observations.

    Parameters:
        filled_lai (numpy.ndarray): Array of filled LAI data.
        lai_time (pandas.Series): Series of timestamps corresponding to LAI data.
        start_year (int): Start year of the data range.
        end_year (int): End year of the data range.

    Returns:
        tuple: Filled LAI data and corresponding dates.
    """
    # Generate all expected MODIS observation dates (every 8 days)
    all_tsteps = []
    for year in range(lai_time.dt.year.min(), lai_time.dt.year.max() + 1):
        year_dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='8D')
        all_tsteps.extend(year_dates)
    
    # Convert to pandas datetime format
    all_time = pd.to_datetime(all_tsteps)
    
    # Check for the presence of all expected dates in the original data
    result = all_time.isin(lai_time)
    
    # Initialize an array for filled LAI data
    temp_array = np.full(result.shape, np.nan)
    
    # Fill gaps in LAI data
    fill_index = 0
    for i, val in enumerate(result):
        if val:
            temp_array[i] = filled_lai[fill_index]
            fill_index += 1
    
    # Reshape the LAI array to exclude incomplete years
    selected_lai = temp_array.reshape(-1, 46)[1:-1]

    # Select dates within the specified range
    selected_dates = all_time[(all_time.year >= start_year) & (all_time.year <= end_year)]
    
    # Interpolate gaps if there are missing values
    if np.isnan(selected_lai).any():
        selected_lai_flatten = selected_lai.flatten()
        positions = np.where(np.isnan(selected_lai_flatten))[0]
        if positions.size > 0:
            for position in positions:
                selected_lai_flatten[position] = (selected_lai_flatten[position - 1] + selected_lai_flatten[position + 1]) / 2
            gap_free_lai = selected_lai_flatten.reshape(-1, 46)
            return gap_free_lai, selected_dates
    else:
        return selected_lai, selected_dates

def rolling_mean(array, window):
    """
    Calculates the rolling mean along a 1D numpy array.

    Parameters:
        array (numpy.ndarray): Input 1D array.
        window (int): Size of the rolling window.

    Returns:
        numpy.ndarray: Array after applying the rolling mean.
    """
    result = np.full_like(array, np.nan)
    for i in range(array.shape[0]):
        start_index = max(0, i - window // 2)
        end_index = min(array.shape[0], i + (window + 1) // 2)
        result[i] = np.mean(array[start_index:end_index], axis=0)
    return result

def smoothing_LAI(gap_free_lai):
    """
    Smooths the gap-free LAI data by calculating climatology, removing mean climatology to obtain anomalies,
    and applying a rolling mean to the anomalies with a window of +/- 6 months. Adds the smoothed anomalies
    back to the climatology to obtain final smoothed LAI values.

    Parameters:
        gap_free_lai (numpy.ndarray): Array containing gap-free LAI data.

    Returns:
        numpy.ndarray: Smoothed LAI values.
    """
    # Calculate climatology by taking the mean of each column
    column_means = np.nanmean(gap_free_lai, axis=0)
    
    # Calculate anomalies by subtracting the climatology from the data
    anomaly = gap_free_lai - column_means
    
    # Apply rolling mean to smooth the anomalies
    anomaly_rolling = rolling_mean(anomaly.flatten(), 13)
    
    # Add the smoothed anomalies back to the climatology
    smoothed_lai = anomaly_rolling + np.tile(column_means, (anomaly.shape[0]))
    
    # Apply additional smoothing using Savitzky-Golay filter
    smoothed_lai_smoothed = savgol_filter(smoothed_lai, window_length=13, polyorder=3)
    
    # Ensure all LAI values are non-negative
    smoothed_lai_smoothed[smoothed_lai_smoothed < 0] = 0
    return smoothed_lai_smoothed

def get_modisLAI_for_station(station_name, time_interval="30min"):
    """
    Retrieves LAI data from MODIS files for a specific station and time range.
    Applies spatial weighting, interpolates missing values, checks data availability, smooths data,
    and resamples to match the resolution of flux tower data.

    Parameters:
        station_name (str): Name of the station.
        time_interval (str): Desired time resolution for resampling (default is "30min").

    Returns:
        pandas.DataFrame: DataFrame containing the resampled and smoothed LAI values.
    """
    # Define the path to MODIS raw data
    modis_path = "/home/khanalp/data/MODIS_Raw/"
    
    # Retrieve MODIS LAI, QC, and SD files for the specified station
    lai_file = [file for file in os.listdir(modis_path) if file.startswith(f"{station_name}_MCD15A2H_Lai_500m_")]
    qc_file = [file for file in os.listdir(modis_path) if file.startswith(f"{station_name}_MCD15A2H_FparLai_QC_")]
    sd_file = [file for file in os.listdir(modis_path) if file.startswith(f"{station_name}_MCD15A2H_LaiStdDev_500m_")]
    
    # Read LAI, QC, and SD data into DataFrames
    df_lai = pd.read_csv(os.path.join(modis_path, lai_file[0]))
    df_sd = pd.read_csv(os.path.join(modis_path, sd_file[0]))
    df_qc = pd.read_csv(os.path.join(modis_path, qc_file[0]))

    # Determine the number of timesteps
    no_tsteps = min(len(df_lai), len(df_sd), len(df_qc)) // max(df_lai['pixel'])

    # Define pixels of interest (center and surrounding pixels)
    pixel_no = [7, 8, 9, 12, 13, 14, 17, 18, 19]
    
    # Extract LAI timestamps
    lai_time = pd.to_datetime(df_lai.loc[df_lai['pixel'] == pixel_no[0], 'calendar_date'])

    # Extract pixel data for LAI, SD, and QC
    lai_pixel = extract_pixel_data(df_lai, no_tsteps=no_tsteps, pixel_no=pixel_no)
    sd_pixel = extract_pixel_data(df_sd, no_tsteps=no_tsteps, pixel_no=pixel_no)
    qc_pixel = extract_pixel_data(df_qc, no_tsteps=no_tsteps, pixel_no=pixel_no)

    # Apply spatial weighting to the LAI data
    weighted_lai_values = get_spatial_weighted_LAI(lai_pixel, sd_pixel, qc_pixel)
        
    # Interpolate missing values in the LAI data
    filled_lai = interpolate_NA_LAI(weighted_lai_values)
    
    # Check data availability and fill any remaining gaps
    gap_free_lai, selected_dates = check_data_availability_LAI(filled_lai, lai_time, start_year=2003, end_year=2023)

    # Smooth the LAI data
    smooth_lai = smoothing_LAI(gap_free_lai)
    
    # Create a DataFrame for the smoothed LAI data
    df_lai_original = pd.DataFrame({'Date': selected_dates, 'LAI': smooth_lai})
    df_lai_original['Date'] = pd.to_datetime(df_lai_original['Date'])
    df_lai_original.set_index('Date', inplace=True)

    # Resample LAI data to match the flux tower resolution
    df_lai_resampled = resampleLAI_to_fluxtower_resolution(data_frame=df_lai_original, resampling_interval=time_interval)

    return df_lai_resampled
