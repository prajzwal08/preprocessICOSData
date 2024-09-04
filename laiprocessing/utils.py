import numpy as np
import pandas as pd
import os
from scipy.signal import savgol_filter
from scipy import interpolate

def list_folders_with_prefix(location, prefix):
    """
    Retrieves a list of folder names within the specified location directory that start with the provided prefix.
    
    Parameters:
        location (str): The directory path where the function will search for folders.
        prefix (str): The prefix that the desired folders should start with.
    
    Returns:
        list: A list of folder names starting with the specified prefix within the given location.
    """
    folders_with_prefix = [folder for folder in os.listdir(location) 
                           if os.path.isdir(os.path.join(location, folder)) and folder.startswith(prefix)]
    return folders_with_prefix

def list_csv_files_in_folder(folder_path, keyword):
    """
    Retrieves a list of file paths for CSV files within the specified folder_path directory that contain the provided keyword in their filenames.
    
    Parameters:
        folder_path (str): The directory path where the function will search for CSV files.
        keyword (str): The keyword that the filenames of desired CSV files should contain.
    
    Returns:
        list: A list of file paths for CSV files containing the specified keyword within the given folder_path.
    """
    csv_files = [os.path.join(folder_path, file) 
                 for file in os.listdir(folder_path) 
                 if file.endswith('.csv') and keyword in file]
    return csv_files

def read_csv_file_with_station_name(station_name, file_paths):
    """
    Reads a CSV file with the specified station name in its filename from a list of file paths.
    
    Parameters:
        station_name (str): The name of the station to search for in the filenames.
        file_paths (list): A list of file paths where CSV files are stored.
        
    Returns:
        pandas.DataFrame: DataFrame containing the data from the CSV file with the specified station name.
        None: If no file with the station name is found.
    """
    for file_path in file_paths:
        if station_name in file_path:
            try:
                df = pd.read_csv(file_path)
                return df
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                return None
    print(f"No file with station name '{station_name}' found.")
    return None

def rolling_mean(array, window):
    """
    Calculates the rolling mean along a 1D numpy array.
    
    Parameters:
        array (numpy.ndarray): Input 1D array.
        window (int): Size of the rolling window.
    
    Returns:
        numpy.ndarray: Resultant array after applying the rolling mean.
    """
    result = np.full_like(array, np.nan)
    for i in range(array.shape[0]):
        start_index = max(0, i - window // 2)
        end_index = min(array.shape[0], i + (window + 1) // 2)
        result[i] = np.mean(array[start_index:end_index], axis=0)
    return result

def smoothing_LAI(gap_free_lai):
    """
    Smoothes the gap-free LAI data by calculating climatology, removing mean climatology to obtain anomalies,
    and applying a rolling mean to the anomalies with a window of +/- 6 months. Finally, it adds the smoothed anomalies
    back to the climatology to obtain smoothed LAI values.
    
    Parameters:
        gap_free_lai (numpy.ndarray): Array containing gap-free LAI data.
    
    Returns:
        numpy.ndarray: Smoothed LAI values.
    """
    # Calculate the mean of each column ignoring NaNs (climatology)
    column_means = np.nanmean(gap_free_lai, axis=0)
    
    # Calculate anomalies by removing mean climatology
    anomaly = gap_free_lai - column_means
    
    # Apply rolling mean to the anomalies
    anomaly_rolling = rolling_mean(anomaly.flatten(), 13)
    
    # Add smoothed anomalies back to the climatology
    smoothed_lai = anomaly_rolling + np.tile(column_means, (anomaly.shape[0]))
    
    # Apply Savitzky-Golay filter for additional smoothing
    smoothed_lai_smoothed = savgol_filter(smoothed_lai, window_length=13, polyorder=3)
    
    # Cap negative values to zero
    smoothed_lai_smoothed[smoothed_lai_smoothed < 0] = 0
    
    return smoothed_lai_smoothed

def interpolate_NA_LAI(unfilled_lai):
    """
    Interpolates missing values (NaNs) in a given array of LAI (Leaf Area Index) using cubic interpolation and caps negative values to zero.
    
    Parameters:
        unfilled_lai (numpy.ndarray): Array containing LAI values with missing values represented as NaNs.
    
    Returns:
        numpy.ndarray: Array with missing values filled using interpolation and negative values capped at zero.
    """
    filled_lai = unfilled_lai.copy()
    
    # Create a mask for NaN values
    nan_mask = np.isnan(unfilled_lai)
    
    # Generate an index array for values
    x = np.arange(len(unfilled_lai))
    
    # Interpolate only at the positions where NaNs are present
    interp_func = interpolate.interp1d(x[~nan_mask], unfilled_lai[~nan_mask], kind='cubic', fill_value="extrapolate")
    
    # Extrapolate the NaN values
    filled_lai[nan_mask] = interp_func(x)[nan_mask]
    
    # Cap negative values to zero
    filled_lai[filled_lai < 0] = 0
    
    # Set the last observation to zero
    filled_lai[-1] = 0
    
    return filled_lai

def resampleLAI_to_fluxtower_resolution(data_frame, resampling_interval='30min'):
    """
    Resamples LAI data to the flux tower resolution.

    Parameters:
        data_frame (pandas.DataFrame): DataFrame containing LAI data with a 'Date' column.
        resampling_interval (str, optional): Resampling interval. Defaults to '30 min'.

    Returns:
        pandas.DataFrame: Resampled and filtered LAI data.
    """
    # Resample to the specified interval and forward fill missing values
    df_filled = data_frame.resample(resampling_interval).ffill()
    
    # Extend the DataFrame index to cover the full range of the year
    start_date_year = df_filled.index[0].year
    last_date_year = df_filled.index[-1].year
    
    start_date_extend = pd.to_datetime(f'{start_date_year}-01-01 00:00:00')
    end_date_extend = pd.to_datetime(f'{last_date_year}-12-31 23:30:00')
    
    # Reindex and forward fill if necessary
    if df_filled.index.max() < end_date_extend:
        df_filled = df_filled.reindex(pd.date_range(start=df_filled.index.min(), end=end_date_extend, freq=resampling_interval)).ffill()
    
    if df_filled.index.min() > start_date_extend:
        # Reindex to include the start_date_extend and backward fill missing values
        df_filled = df_filled.reindex(pd.date_range(start=start_date_extend, end=df_filled.index.max(), freq=resampling_interval)).bfill()
    
    return df_filled
