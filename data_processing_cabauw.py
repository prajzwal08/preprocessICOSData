import os
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List, Union
from utils import add_lai_data_to_xarray, add_station_info_to_xarray, update_co2_data, update_vpd_data

def extract_date(file_name: str) -> Optional[datetime]:
    """
    Extract the date from the file name.

    Parameters:
        file_name (str): The name of the file.

    Returns:
        Optional[datetime]: Extracted date in datetime format, or None if parsing fails.
    """
    try:
        date_str = file_name.split("_v1.0_")[1].split(".")[0]
        return datetime.strptime(date_str, "%Y%m")
    except (IndexError, ValueError):
        return None

def get_combined_meteorological_dataset(
    cabauw_file_location: str, 
    meteorological_file_keyword: str
) -> Optional[xr.Dataset]:
    """
    Combine meteorological datasets based on a keyword found in file names.

    Parameters:
        cabauw_file_location (str): Path to the directory containing the files.
        meteorological_file_keyword (str): Keyword to filter meteorological files.

    Returns:
        Optional[xr.Dataset]: Combined dataset after processing, or None if no datasets found.
    """
    # List all files in the directory that contain the keyword
    meteorological_files = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(cabauw_file_location)
        for file in files
        if meteorological_file_keyword in file
    ]

    # Sort the files based on the extracted date
    sorted_meteorological_files = sorted(meteorological_files, key=extract_date)

    # Initialize a list to store datasets
    cleaned_datasets = []

    # Process each file
    for file in sorted_meteorological_files:
        # Open the dataset
        ds = xr.open_dataset(file)

        # Drop 'day_in_time_interval' dimension if it exists
        if 'day_in_time_interval' in ds.dims:
            print(f"Dropping 'day_in_time_interval' from dataset {file}")
            ds_cleaned = ds.drop_dims('day_in_time_interval')
        else:
            ds_cleaned = ds

        cleaned_datasets.append(ds_cleaned)
        ds.close()

    # Check if datasets are available to combine
    if not cleaned_datasets:
        print("No datasets found to combine.")
        return None

    # Concatenate all cleaned datasets along the 'time' dimension
    combined_dataset_meteorology = xr.concat(cleaned_datasets, dim='time')

    return combined_dataset_meteorology

def select_variables(dataset: xr.Dataset, variables: List[str]) -> xr.Dataset:
    """
    Select specified variables from a dataset.

    Parameters:
        dataset (xr.Dataset): The original xarray dataset.
        variables (List[str]): List of variable names to select.

    Returns:
        xr.Dataset: New dataset containing only the selected variables.
    """
    # Ensure the dataset contains all requested variables
    missing_vars = [var for var in variables if var not in dataset]
    if missing_vars:
        raise ValueError(f"Variables not found in the dataset: {missing_vars}")
    
    return dataset[variables]

def combine_datasets(dataset1: xr.Dataset, dataset2: xr.Dataset) -> xr.Dataset:
    """
    Combine two datasets into one, aligning on all common dimensions.

    Parameters:
        dataset1 (xr.Dataset): First xarray dataset.
        dataset2 (xr.Dataset): Second xarray dataset.

    Returns:
        xr.Dataset: Combined dataset.
    """
    combined = xr.merge([dataset1, dataset2])
    return combined

def resample_and_aggregate(dataset: xr.Dataset, freq: str = '30min', sum_variable: str = 'RAIN') -> xr.Dataset:
    """
    Resample the dataset to the specified frequency and apply aggregation functions.

    Parameters:
        dataset (xr.Dataset): The xarray dataset to resample.
        freq (str): Frequency for resampling (e.g., '30min').
        sum_variable (str): Variable to apply sum aggregation.

    Returns:
        xr.Dataset: Resampled and aggregated dataset.
    """
    # Separate datasets for mean and sum variables
    mean_vars = [var for var in dataset.data_vars if var != sum_variable]
    sum_vars = [sum_variable] if sum_variable in dataset.data_vars else []

    # Apply mean aggregation to all variables except the sum_variable
    ds_mean = dataset[mean_vars].resample(time=freq).mean(skipna=True)
    # Apply sum aggregation only to the specified variable
    ds_sum = dataset[sum_vars].resample(time=freq).sum() if sum_vars else xr.Dataset()

    # Merge the mean and sum datasets
    combined_dataset = xr.merge([ds_mean, ds_sum])

    return combined_dataset

def convert_units(data_xarray: xr.Dataset) -> xr.Dataset:
    """
    Converts units of specific variables in the xarray dataset to match the desired format.

    Parameters:
        data_xarray (xr.Dataset): The xarray dataset to update.

    Returns:
        xr.Dataset: Updated xarray dataset with converted units.
    """
    # Convert precipitation from mm/30 min to mm/s
    data_xarray['Precip'] = data_xarray['Precip'] / (30 * 60)
    
    # Convert surface pressure from hPa to Pa
    data_xarray['Psurf'] = data_xarray['Psurf'] * 100
    
    return data_xarray

def get_closest_index(target_time: pd.Timestamp, time_pd: pd.DatetimeIndex) -> int:
    """
    Find the index of the closest time in time_pd to the target_time.

    Parameters:
        target_time (pd.Timestamp): The target time to find the closest match for.
        time_pd (pd.DatetimeIndex): The list of times to search through.

    Returns:
        int: The index of the closest time in time_pd.
    """
    # Calculate the absolute differences between the target_time and each time in time_pd
    delta = np.abs(time_pd - target_time)
    
    # Find the index of the minimum difference
    closest_index = delta.argmin()
    
    return closest_index


def fill_missing_values(data_xarray: xr.Dataset, variable_name: str) -> xr.Dataset:
    """
    Fill missing values in a specified variable using interpolation based on the nearest available data.

    Parameters:
        data_xarray (xr.Dataset): The xarray dataset with missing values.
        variable_name (str): The variable name in the dataset to fill missing values for.

    Returns:
        xr.Dataset: Updated xarray dataset with missing values filled.
    """
    # Extract time and data
    time = data_xarray['time']
    data = data_xarray[variable_name].values.flatten()
    
    # Identify the missing times
    missing_times = time.values[np.isnan(data)]
    
    # Convert times to pandas datetime for easier manipulation
    time_pd = pd.to_datetime(time.values)
    
    # Initialize a new array for filled values
    filled_values = np.copy(data)
    
    for missing_time in missing_times:
        missing_time_pd = pd.to_datetime(missing_time)
        prev_day = missing_time_pd - pd.DateOffset(days=1)
        next_day = missing_time_pd + pd.DateOffset(days=1)
        
        prev_index = get_closest_index(prev_day, time_pd)
        next_index = get_closest_index(next_day, time_pd)
        
        prev_value = data[prev_index] if not np.isnan(data[prev_index]) else np.nan
        next_value = data[next_index] if not np.isnan(data[next_index]) else np.nan
        
        valid_values = [v for v in [prev_value, next_value] if not np.isnan(v)]
        mean_value = np.nanmean(valid_values) if valid_values else np.nan
        
        missing_index = np.where(time.values == missing_time)[0][0]
        filled_values[missing_index] = mean_value
    
    original_shape = data_xarray[variable_name].shape
    filled_values_reshaped = filled_values.reshape(original_shape)
    
    filled_data_xarray = data_xarray.copy(deep=True)
    filled_data_xarray[variable_name].values = filled_values_reshaped
    
    return filled_data_xarray

def find_vars_with_missing_values(data_xarray: Union[xr.DataArray, xr.Dataset]) -> List[str]:
    """
    Find variables with missing values in an xarray DataArray or Dataset,
    excluding those with '_qc' in their names.

    Parameters:
        data_xarray (Union[xr.DataArray, xr.Dataset]): The xarray object to check.

    Returns:
        List[str]: List of variable names with missing values.
    """
    # Filter variables to exclude those with '_qc' in their names
    data_vars_without_qc = [var_name for var_name in data_xarray.data_vars if '_qc' not in var_name]

    variables_with_missing_values = []

    for var_name in data_vars_without_qc:
        variable = data_xarray[var_name]
        if variable.isnull().any():
            variables_with_missing_values.append(var_name)
    
    return variables_with_missing_values


if __name__ == "__main__":
    # Define file paths
    cabauw_file_location = "/home/khanalp/data/fluxsites_NL/incoming/cabauw"
    cams_path = "/home/khanalp/data/cams/cams_europe_2003_2020.nc"
    lai_modis_path = "/home/khanalp/data/processed/lai/modis"
    output_directory = '/home/khanalp/data/processed/input_pystemmus'
    station_details = '/home/khanalp/code/PhD/preprocessICOSdata/csvs/02_station_with_elevation_heightcanopy.csv'

    # Define the output file path
    combined_dataset_30mins_path = os.path.join(cabauw_file_location, "intermediate", "combined_dataset_30mins.nc")

    # Process and save datasets if not already saved
    if not os.path.exists(combined_dataset_30mins_path):
        meteorological_file_keyword = "cesar_surface_meteo"
        radiation_file_keyword = "cesar_surface_radiation"
        combined_dataset_meteorology = get_combined_meteorological_dataset(cabauw_file_location, meteorological_file_keyword)
        combined_dataset_radiation = get_combined_meteorological_dataset(cabauw_file_location, radiation_file_keyword)

        # Load station details from CSV
        df_station_details = pd.read_csv(station_details)
        station_name = "NL-Cab"
        station_info = df_station_details[df_station_details['station_name'] == station_name]

        # Print long_name attributes for radiation variables
        for var_name in combined_dataset_radiation.data_vars:
            var = combined_dataset_radiation[var_name]
            if 'long_name' in var.attrs:
                print(f"{var_name}: {var.attrs['long_name']}")
            else:
                print(f"{var_name}: 'long_name' attribute not found")
        
        # Select and combine variables from meteorological and radiation datasets
        variables_from_meteorology = ['TA002', 'P0', 'RAIN', 'F010', 'RH002', 'Q002']
        variables_from_radiation = ['SWD', 'LWD']

        dataset_meteorology_selected = select_variables(combined_dataset_meteorology, variables_from_meteorology)
        dataset_radiation_selected = select_variables(combined_dataset_radiation, variables_from_radiation)
        combined_dataset = combine_datasets(dataset_meteorology_selected, dataset_radiation_selected)

        # Resample and aggregate dataset
        combined_dataset_30mins = resample_and_aggregate(combined_dataset, freq='30min', sum_variable='RAIN')
        combined_dataset_30mins.to_netcdf(combined_dataset_30mins_path)

    # Define alternative start and end dates
    alternative_start_date = np.datetime64('2003-01-01 00:00:00')
    alternative_end_date = np.datetime64('2020-12-31 00:00:00')

    combined_dataset_30mins = xr.open_dataset(combined_dataset_30mins_path)
    # Filter dataset based on available time range
    data_xarray = combined_dataset_30mins.sel(time=slice(
        alternative_start_date if combined_dataset_30mins.time.values.min() < alternative_start_date else None,
        alternative_end_date if combined_dataset_30mins.time.values.max() > alternative_end_date else None
    ))

    # Rename variables according to mapping
    rename_mapping = {
        'TA002': 'Tair',
        'P0': 'Psurf',
        'F010': 'Wind',
        'RH002': 'RH',
        'Q002': 'Qair',
        'SWD': 'SWdown',
        'LWD': 'LWdown',
        'RAIN': 'Precip'
    }
    data_xarray = data_xarray.rename(rename_mapping)

    # Expand dimensions to include 'x' and 'y' coordinates with dummy values
    data_xarray = data_xarray.expand_dims({'x': [1], 'y': [2]})
    data_xarray['x'] = data_xarray['x'].astype('float64')
    data_xarray['y'] = data_xarray['y'].astype('float64')

    # Add station-specific information, MODIS LAI data, and update CO2 and VPD data
    data_xarray = add_station_info_to_xarray(data_xarray, station_info)
    data_xarray = add_lai_data_to_xarray(data_xarray, lai_modis_path, station_name)
    data_xarray = update_co2_data(data_xarray=data_xarray, cams_path=cams_path)
    data_xarray = update_vpd_data(data_xarray=data_xarray)

    # Convert units and ensure variables are in float32 format
    data_xarray = convert_units(data_xarray)
    
    #Data format conversion
    for var_name in data_xarray.data_vars:
        if var_name not in ['IGBP_veg_short', 'IGBP_veg_long']:
            data_xarray[var_name] = data_xarray[var_name].astype('float32')

    # Fill missing values if any
    missing_vars = find_vars_with_missing_values(data_xarray)
    if missing_vars:
        for variable in missing_vars:
            data_xarray = fill_missing_values(data_xarray, variable)

    # Extract start and end years and construct the filename for saving
    time_pd = pd.to_datetime(data_xarray.time.values)
    start_year = time_pd.min().year
    end_year = time_pd.max().year
    filename = f"FLX_{station_name}_FLUXNET2015_FULLSET_{start_year}-{end_year}.nc"
    output_path = os.path.join(output_directory, filename)

    # Save the xarray dataset to NetCDF file
    data_xarray.to_netcdf(output_path)
    print("Dataset saved successfully.")
