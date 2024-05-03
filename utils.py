import os 
import pandas as pd
import xarray as xr
import numpy as np

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
                return None


def select_rename_convert_to_xarray(data_frame, selected_variables, rename_mapping):
    """
    Selects required variables from ICOS data, renames them, and converts to xarray dataset.

    Parameters:
        data_frame (pandas.DataFrame): DataFrame containing ICOS data.
        selected_variables (list): List of variables to select.
        rename_mapping (dict): Dictionary containing renaming mapping for variables.

    Returns:
        xarray.Dataset: Processed ICOS data as xarray dataset.
    """
    # Initialize an empty DataFrame
    df_selected = pd.DataFrame()
    
    # Iterate over selected variables
    for var in selected_variables:
        # Check if the variable exists in the DataFrame
        if var in data_frame.columns:
            # Rename and select the variable
            df_selected[var] = data_frame[var]
        else:
            #print(f"Variable '{var}' not found in the DataFrame. Adding it with missing values (-9999).")
            # Add the variable with missing values (-9999)
            df_selected[var] = np.full_like(data_frame.index, -9999)
    
    # Rename columns
    df_selected = df_selected.rename(columns=rename_mapping)
    
    # Make xarray dataset
    xds = xr.Dataset.from_dataframe(df_selected)
    xds_indexed = xds.assign_coords(index=pd.to_datetime(xds['TIMESTAMP_START'], format='%Y%m%d%H%M'))
    xds_indexed = xds_indexed.rename({'index':'time'})

    # Adding x,y to the dimensions
    xds_dimension = xds_indexed.expand_dims({'x': [1], 'y': [2]})
    
    # Dropping variable 'TIMESTAMP_START' because it's already indexed
    xds_dimension = xds_dimension.drop_vars('TIMESTAMP_START')
    
    # Converting x, y to float64
    xds_dimension['x'] = xds_dimension['x'].astype('float64')
    xds_dimension['y'] = xds_dimension['y'].astype('float64')
    
    return xds_dimension

def replace_negative_with_mean_of_nearest(arr):
    """
    Replaces negative values in the input array with the mean of the nearest non-negative values.
    
    Parameters:
        arr (numpy.ndarray): Input array.
    
    Returns:
        numpy.ndarray: Array with negative values replaced by the mean of the nearest non-negative values.
    """
    neg_indices = np.where(arr < 0)[0]  # Get indices where values are less than zero
    for i in neg_indices:
        # Find nearest non-negative values before and after the negative value
        left_index = i - 1
        while left_index in neg_indices:
            left_index -= 1
        right_index = i + 1
        while right_index in neg_indices:
            right_index += 1
    
        # Replace negative value with the mean of the nearest non-negative values
        arr[i] = np.mean([arr[left_index], arr[right_index]])
    
    return arr

def calculate_relative_and_specific_humidity(vpd, Tair, pressure):
    """
    Calculate relative humidity (RH) and specific humidity (q) from input vpd, Tair, and pressure.
    
    Parameters:
    vpd (array-like): Array of vapor pressure deficit values. (in hPa)
    Tair (array-like): Array of air temperature values (in Celsius).
    pressure (array-like): Array of air pressure values (in KPa).
    
    Returns:
    RH (array-like): Array of relative humidity values. (%)
    q (array-like): Array of specific humidity values. ()
    """
    
    # Gas constant of dry air and vapor.
    Rd = 287.058
    Rv = 461.5
     
    # Calculating saturation vapor pressure (es) from the Tair.
    es = 0.6108 * np.exp((17.27 * Tair) / (Tair + 237.3)) * 10 #mutilplied by 10 to get hPa from kPa.
    
    # Calculating the actual vapor pressure from es and vpd and then RH.
    ea = es - vpd
    RH = (ea / es) * 100
    #some values are negative so. 
    RH_modified = replace_negative_with_mean_of_nearest(RH)
    
    # Calculating the specific humidity (q) from ea.
    w = ea * Rd / (Rv * (pressure * 10 - ea))  # Since pressure in KPa, ea in hPa, same as VPD.
    q = w / (w + 1)
    qair_modified = replace_negative_with_mean_of_nearest(q)
    
    return RH_modified, qair_modified

# Function to interpolate missing values with linear interpolation if missing values are less than 10%
def interpolate_missing_values(data_array):
    # Count the number of missing values (-9999)
    missing_count = np.sum(data_array == -9999)
    total_count = data_array.size
    # Calculate the percentage of missing values
    missing_percentage = missing_count / total_count * 100

    if missing_percentage < 10:
        # Perform linear interpolation to fill missing values
        data_array = data_array.where(data_array != -9999).interpolate_na(dim='time')

    return data_array

# Function to check missing values within each month individually

def check_missing_values_monthly(data_array):
    # Group the data by month
    monthly_data = data_array.groupby('time.month')
    
    missing_perecentage = []
    # Iterate over each month
    for month, data in monthly_data:
        # Count the number of missing values (-9999) for each month
        missing_count = np.sum(data == -9999)
        total_count = data.size
        # Calculate the percentage of missing values for each month
        missing_perecentage.append((missing_count / total_count * 100).values)
    return missing_perecentage
    

        
