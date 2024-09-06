"""
Purpose: This script preprocesses ICOS data, including selecting, renaming, and converting variables,
calculating additional variables like LAI, CO2, relative humidity, and specific humidity, 
and saving the processed data to NetCDF files which is input for pystemmusscope.
"""

# importing libraries
import pandas as pd
import xarray as xr
import numpy as np
import os
from utils import read_csv_file_with_station_name
from utils import calculate_relative_and_specific_humidity
from getco2 import get_co2_data

def select_rename_convert_to_xarray(data_frame, selected_variables, rename_mapping):
    """
    Selects required variables from ICOS data, renames them, and converts to xarray dataset.

    Parameters:
        data_frame (pandas.DataFrame): DataFrame containing ICOS data.
        selected_variables (list): List of variables to select from the DataFrame.
        rename_mapping (dict): Dictionary containing renaming mapping for variables.

    Returns:
        xarray.Dataset: Processed ICOS data as an xarray dataset.
    """
    # Initialize an empty DataFrame to hold selected variables
    df_selected = pd.DataFrame()
    
    # Iterate over the list of selected variables
    for var in selected_variables:
        # Check if the variable exists in the input DataFrame
        if var in data_frame.columns:
            # If the variable exists, add it to the df_selected DataFrame
            df_selected[var] = data_frame[var]
        else:
            # If the variable does not exist, add it with missing values (-9999)
            df_selected[var] = np.full_like(data_frame.index, -9999)
            # Optional: Uncomment the line below to print a warning message
            # print(f"Variable '{var}' not found in the DataFrame. Adding it with missing values (-9999).")
    
    # Rename columns according to the provided mapping
    df_selected = df_selected.rename(columns=rename_mapping)
    
    # Convert the DataFrame to an xarray Dataset
    xds = xr.Dataset.from_dataframe(df_selected)
    
    # Assign datetime coordinates to the 'index' dimension based on the 'TIMESTAMP_START' column
    xds_indexed = xds.assign_coords(index=pd.to_datetime(xds['TIMESTAMP_START'], format='%Y%m%d%H%M'))
    
    # Rename the 'index' dimension to 'time'
    xds_indexed = xds_indexed.rename({'index': 'time'})
    
    # Expand dimensions to include 'x' and 'y' coordinates with dummy values
    xds_dimension = xds_indexed.expand_dims({'x': [1], 'y': [2]})
    
    # Drop the 'TIMESTAMP_START' variable as it is no longer needed
    xds_dimension = xds_dimension.drop_vars('TIMESTAMP_START')
    
    # Ensure 'x' and 'y' coordinates are of type float64 for consistency
    xds_dimension['x'] = xds_dimension['x'].astype('float64')
    xds_dimension['y'] = xds_dimension['y'].astype('float64')
    
    # Return the final xarray Dataset with selected, renamed variables and added dimensions
    return xds_dimension


def add_station_info_to_xarray(data_xarray, station_info):
    #getting other  variables in xarray.
    data_xarray['latitude'] = xr.DataArray(np.array(station_info['latitude']).reshape(1,-1), dims=['x','y'])
    data_xarray['longitude'] = xr.DataArray(np.array(station_info['longitude']).reshape(1,-1), dims=['x','y'])
    data_xarray['reference_height'] = xr.DataArray(np.array(station_info['measurement_height']).reshape(1,-1), dims=['x','y'])
    data_xarray['canopy_height'] = xr.DataArray(np.array(station_info['height_canopy_field_information']).reshape(1,-1), dims=['x','y'])
    data_xarray['elevation'] = xr.DataArray(np.array(station_info['elevation']).reshape(1,-1), dims=['x','y'])
    data_xarray['IGBP_veg_short'] = xr.DataArray(np.array(station_info['IGBP_short_name'], dtype = 'S200').reshape(1,-1),dims = ['x','y'])
    data_xarray['IGBP_veg_long'] = xr.DataArray(np.array(station_info['IGBP_long_name'], dtype = 'S200').reshape(1,-1),dims = ['x','y'])
    return data_xarray

def add_lai_data_to_xarray(data_xarray, lai_modis_path, station_name):
    """
    Finds the LAI file, reads and filters LAI data, and adds it to the xarray dataset.

    Parameters:
    - data_xarray (xr.Dataset): The xarray dataset to which LAI data will be added.
    - lai_modis_path (str): Directory path containing the LAI CSV files.
    - station_name (str): The name of the station to match the LAI file.

    Returns:
    - xr.Dataset: Updated xarray dataset with added LAI data.
    """
    # Find the LAI file that matches the station_name
    file_path = next((os.path.join(lai_modis_path, file) for file in os.listdir(lai_modis_path) if station_name in file), None)
    
    if file_path is None:
        raise FileNotFoundError(f"No LAI file found for station: {station_name}")
    
    # Read LAI data
    df_lai = pd.read_csv(file_path)
    df_lai.columns = ['Date', 'LAI']
    df_lai['Date'] = pd.to_datetime(df_lai['Date'])
    
    # Filter LAI data based on the time range in data_xarray
    start_date = pd.to_datetime(data_xarray['time'].values.min())
    end_date = pd.to_datetime(data_xarray['time'].values.max())
    df_lai_filtered = df_lai[(df_lai['Date'] >= start_date) & (df_lai['Date'] <= end_date)]
    
    # Check if the lengths match
    if len(df_lai_filtered) != len(data_xarray['time']):
        raise ValueError("Length mismatch between LAI data and xarray time dimension")
    
    # Add LAI data to data_xarray
    data_xarray['LAI'] = xr.DataArray(df_lai_filtered['LAI'].values.reshape(1, 1, -1), dims=['x', 'y', 'time'])
    data_xarray['LAI_alternative'] = xr.DataArray(df_lai_filtered['LAI'].values.reshape(1, 1, -1), dims=['x', 'y', 'time'])
    return data_xarray

def update_co2_data(data_xarray, counts, cams_path):
    """
    Updates the 'CO2air' data in the xarray dataset using CAMS data if missing.

    Parameters:
    - data_xarray (xr.Dataset): The xarray dataset to update.
    - counts (dict): Dictionary containing the counts of missing values for different variables.
    - cams_path (str): File path to the CAMS data.

    Returns:
    - xr.Dataset: Updated xarray dataset with 'CO2air' data.
    """
    # Check if CO2air is missing
    if counts['CO2air'] > 0:
        # Get CO2 data from CAMS
        co2_array = get_co2_data(
            latitude=data_xarray['latitude'].values.flatten(),
            longitude=data_xarray['longitude'].values.flatten(),
            start_time=data_xarray.time.values.min(),
            end_time=data_xarray.time.values.max(),
            resampling_interval="30 min",
            file_path=cams_path
        )
        # Update 'CO2air' in data_xarray with CAMS data
        data_xarray['CO2air'] = xr.DataArray(co2_array.reshape(1, 1, -1), dims=['x', 'y', 'time'])
        # Set attributes to indicate the source of the data
        attributes_CO2air = {'method': 'from cams, due to insufficient field record'}
        data_xarray['CO2air'].attrs.update(attributes_CO2air)
    else:
        # Set attributes if 'CO2air' data is from the field
        attributes_CO2air = {'method': 'from field'}
        data_xarray['CO2air'].attrs.update(attributes_CO2air)

    return data_xarray

def update_humidity_data(data_xarray, counts):
    """
    Updates the 'RH' (Relative Humidity) and 'Qair' (Specific Humidity) in the xarray dataset.
    
    Parameters:
    - data_xarray (xr.Dataset): The xarray dataset to update.
    - counts (dict): Dictionary containing the counts of missing values for different variables.
    
    Returns:
    - xr.Dataset: Updated xarray dataset with 'RH' and 'Qair' data.
    """
    # Calculate relative and specific humidity
    relative_humidity, specific_humidity = calculate_relative_and_specific_humidity(
        data_xarray.VPD.values.flatten(), 
        data_xarray.Tair.values.flatten(), 
        data_xarray.Psurf.values.flatten()
    )
    
    # Check if RH data is missing
    if counts['RH'] > 0:
        # If missing, replace with calculated values
        data_xarray['Qair'] = xr.DataArray(specific_humidity.reshape(1, 1, -1), dims=['x', 'y', 'time'])
        data_xarray['RH'] = xr.DataArray(relative_humidity.reshape(1, 1, -1), dims=['x', 'y', 'time'])
        # Update attributes to indicate the source of the data
        attributes_RH_Qair = {'method': 'calculated from VPD, Tair, Psurf, ignoring field data.'}
        data_xarray['Qair'].attrs.update(attributes_RH_Qair)
        data_xarray['RH'].attrs.update(attributes_RH_Qair)
    else:
        # If RH data is not missing, only update Qair with calculated values
        data_xarray['Qair'] = xr.DataArray(specific_humidity.reshape(1, 1, -1), dims=['x', 'y', 'time'])
        # Update attributes for Qair
        attributes_RH_Qair = {'method': 'calculated from VPD, Tair, Psurf, ignoring field data.'}
        data_xarray['Qair'].attrs.update(attributes_RH_Qair)
    
    return data_xarray


def convert_units(data_xarray):
    """
    Converts units of specific variables in the xarray dataset to match the desired format.
    
    Parameters:
    - data_xarray (xr.Dataset): The xarray dataset to update.
    
    Returns:
    - xr.Dataset: Updated xarray dataset with converted units.
    """
    # Convert precipitation from mm/30 min to mm/s
    data_xarray['Precip'] = data_xarray['Precip'] / (30 * 60)
    
    # Convert air temperature from degrees Celsius to Kelvin
    data_xarray['Tair'] = data_xarray['Tair'] + 273.15
    
    # Convert surface pressure from kPa to Pa
    data_xarray['Psurf'] = data_xarray['Psurf'] * 1000
    
    return data_xarray

if __name__ == "__main__":   
    # Define file paths
    modis_path = "/home/khanalp/data/MODIS_Raw/"
    cams_path = "/home/khanalp/data/cams/cams_europe_2003_2020.nc"
    ICOS_location = "/home/khanalp/data/ICOS2020"
    lai_modis_path = "/home/khanalp/data/processed/lai/modis"
    output_directory = '/home/khanalp/data/processed/input_pystemmus'
    prefix = "FLX"
    station_details = '/home/khanalp/code/PhD/preprocessICOSdata/csvs/02_station_with_elevation_heightcanopy.csv'
    
    # Define alternative start date (2003) due to data availability constraints
    alternative_start_date = np.datetime64('2003-01-01 00:00:00')

    # Load station details from CSV
    df_station_details = pd.read_csv(station_details)

    # Define the list of variables to select from ICOS data
    selected_variables = [
        'TIMESTAMP_START',
        'TA_F',
        'TA_F_QC',
        'SW_IN_F',
        'SW_IN_F_QC',
        'LW_IN_F',
        'LW_IN_F_QC',
        'VPD_F',
        'VPD_F_QC',
        'PA_F',
        'PA_F_QC',
        'P_F',
        'P_F_QC',
        'WS_F',
        'WS_F_QC',
        'RH',
        'CO2_F_MDS',
        'CO2_F_MDS_QC'
    ]

    # Define the mapping for renaming variables
    rename_mapping = {
        'TA_F': 'Tair',
        'TA_F_QC': 'Tair_qc',
        'SW_IN_F': 'SWdown',
        'SW_IN_F_QC': 'SWdown_qc',
        'LW_IN_F': 'LWdown',
        'LW_IN_F_QC': 'LWdown_qc',
        'VPD_F': 'VPD',
        'VPD_F_QC': 'VPD_qc',
        'PA_F': 'Psurf',
        'PA_F_QC': 'Psurf_qc',
        'P_F': 'Precip',
        'P_F_QC': 'Precip_qc',
        'WS_F': 'Wind',
        'WS_F_QC': 'Wind_qc',
        'CO2_F_MDS': 'CO2air',
        'CO2_F_MDS_QC': 'CO2air_qc'
    }

    # Process data for each station listed in the station details CSV
    for index, station_info in df_station_details.iterrows():
        station_name = str(station_info['station_name'])
        # Skip processing if the output file for the station already exists
        if any(station_name in filename for filename in os.listdir(output_directory)):
            print(f"Output file already exists for station {station_name}. Skipping...")
            continue
        
        print(f"Processing starts for station {station_name}")
        
        try:
            # Read the corresponding CSV file for the ICOS station in fluxnet format
            df_insitu = read_csv_file_with_station_name(ICOS_location, station_name)
            
            if df_insitu is not None:
                # Filter for specific station info
                station_info = df_station_details[df_station_details['station_name'] == station_name]
            
                # Convert the DataFrame to xarray, selecting and renaming variables
                data_xarray = select_rename_convert_to_xarray(df_insitu, selected_variables, rename_mapping)
                
                # Add station-specific information to the xarray dataset
                data_xarray = add_station_info_to_xarray(data_xarray, station_info)
                
                # Count the number of missing values (denoted by -9999) in each variable
                counts = {var: np.count_nonzero(data_xarray[var].values == -9999) for var in data_xarray.data_vars}
                
                # If the start date is before the alternative start date, filter the dataset
                if data_xarray.time.values.min() < alternative_start_date:
                    data_xarray = data_xarray.sel(time=slice(alternative_start_date, None))
                
                # Add MODIS LAI data to the xarray dataset
                data_xarray = add_lai_data_to_xarray(data_xarray, lai_modis_path, station_name)
                
                # Update CO2 data in the xarray dataset
                data_xarray = update_co2_data(data_xarray, counts, cams_path)
                
                # Update RH and Qair data in the xarray dataset
                data_xarray = update_humidity_data(data_xarray, counts)

                # Convert units in the xarray dataset
                data_xarray = convert_units(data_xarray)

                # Ensure all variables are in float32 format except for specific ones
                for var_name in data_xarray.data_vars:
                    if var_name not in ['IGBP_veg_short', 'IGBP_veg_long']:
                        data_xarray[var_name] = data_xarray[var_name].astype('float32')
                
                # Check for missing values in variables (excluding those with '_qc')
                data_vars_without_qc = [var_name for var_name in data_xarray.data_vars if '_qc' not in var_name]
                no_missing_values = True
                
                # Print missing value status for each variable
                for var_name in data_vars_without_qc:
                    has_missing_values = data_xarray[var_name].isnull().any()
                    if has_missing_values:
                        print(f"Variable '{var_name}' has missing values.")
                        no_missing_values = False
                        break

                # Save the dataset if there are no missing values
                if no_missing_values:
                    # Extract start and end years from the time coordinates
                    time_pd = pd.to_datetime(data_xarray.time.values)
                    start_year = time_pd.min().year
                    end_year = time_pd.max().year
                    
                    # Construct the filename for saving the dataset
                    filename = f"FLX_{station_name}_FLUXNET2015_FULLSET_{start_year}-{end_year}.nc"
                    output_path = os.path.join(output_directory, filename)
                    
                    # Save the xarray dataset to NetCDF file
                    data_xarray.to_netcdf(output_path)
                    print("Dataset saved successfully.")
                else:
                    print("There are missing values in one or more variables. Dataset not saved.")

                pass  # Replace with actual processing code
            
            else:
                print(f"No data found for station {station_name}. Skipping...")
        
        except Exception as e:
            print(f"An error occurred while processing station {station_name}: {e}")
            continue
        
        