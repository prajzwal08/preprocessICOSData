import numpy as np
import pandas as pd
import os

def list_folders_with_prefix(location, prefix):


    """
    Retrieves a list of folder names within the specified location directory that start with the provided prefix.
    
    Parameters:
        location (str): The directory path where the function will search for folders.
        prefix (str): The prefix that the desired folders should start with.
    
    Returns:
        list: A list of folder names starting with the specified prefix within the given location.
    """
    folders_with_prefix = [folder for folder in os.listdir(location) if os.path.isdir(os.path.join(location, folder)) and folder.startswith(prefix)]
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
    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv') and keyword in file]
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
    
    
def resampleLAI_to_fluxtower_resolution(data_frame,  resampling_interval='30min'):
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
    start_date_year = df_filled.index[0].year
    last_date_year = df_filled.index[-1].year
    
    # Reindex to extend the index until the end of the last year and forward fill
    start_date_extend = pd.to_datetime(f'{start_date_year}-01-01 00:00:00')
    end_date_extend = pd.to_datetime(f'{last_date_year}-12-31 23:30:00')
    
    if df_filled.index.max() < end_date_extend:
        df_filled = df_filled.reindex(pd.date_range(start=df_filled.index.min(), end=end_date_extend, freq=resampling_interval)).ffill()
    
    if df_filled.index.min() > start_date_extend:
        # If not, reindex the DataFrame to include start_date_extend and forward fill missing values
        df_filled = df_filled.reindex(pd.date_range(start=start_date_extend, end=df_filled.index.max(), freq=resampling_interval)).bfill()
        
    # Filter DataFrame based on start_date and end_date
    #filtered_df = df_filled[(df_filled.index >= start_date) & (df_filled.index <= end_date)]

    return df_filled