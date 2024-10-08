import os 
import pandas as pd
import xarray as xr
import numpy as np
from typing import Union,Optional,Tuple

# def list_folders_with_prefix(location):
#     """
#     Retrieves a list of folder names within the specified location directory that start with the provided prefix.
    
#     Parameters:
#         location (str): The directory path where the function will search for folders.
#         prefix (str): The prefix that the desired folders should start with.
    
#     Returns:
#         list: A list of folder names starting with the specified prefix within the given location.
#     """
#     folders_with_prefix = [folder for folder in os.listdir(location) if os.path.isdir(os.path.join(location, folder)) and folder.startswith(prefix)]
#     return folders_with_prefix

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

def read_csv_file_with_station_name(location, station_name):
    """
    Reads a CSV file with the specified station name in its filename from a list of file paths.
    
    Parameters:
        station_name (str): The name of the station to search for in the filenames.
        file_paths (list): A list of file paths where CSV files are stored.
        
    Returns:
        pandas.DataFrame: DataFrame containing the data from the CSV file with the specified station name.
        None: If no file with the station name is found.
    """
     # List folders with certain prefix.
    folders =  [folder for folder in os.listdir(location) if os.path.isdir(os.path.join(location, folder))]

    #Inside folder find the csv files 
    csv_files = []
    for folder in folders:
        folder_path = os.path.join(location, folder)
        csv_files.extend(list_csv_files_in_folder(folder_path, "FULLSET_HH"))

    for file_path in csv_files:
        if station_name in file_path:
            try:
                df = pd.read_csv(file_path)
                return df
            except Exception:
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

def calculate_vpd(RH: Union[np.ndarray, list, float], Tair: Union[np.ndarray, list, float]) -> Union[np.ndarray, float]:
    """
    Calculate vapor pressure deficit (VPD) from relative humidity (RH) and air temperature (Tair).
    
    Parameters:
    RH (Union[np.ndarray, list, float]): Relative humidity values (in %). Can be a numpy array, list, or single float.
    Tair (Union[np.ndarray, list, float]): Air temperature values (in Celsius). Can be a numpy array, list, or single float.
    
    Returns:
    Union[np.ndarray, float]: Vapor pressure deficit values (in hPa). Will be a numpy array if inputs are arrays/lists, or a float if inputs are single values.
    """
    
    # Convert inputs to numpy arrays for consistent operations
    Tair = Tair-273.15
    RH = np.clip(RH,0,100)
    
    # Calculate the saturation vapor pressure (es) from Tair.
    es = 0.6108 * np.exp((17.27 * Tair) / (Tair + 237.3)) * 10  # Convert from kPa to hPa.
    
    # Calculate actual vapor pressure (ea) from RH and es.
    ea = (RH / 100) * es 
    
    # Calculate vapor pressure deficit (VPD).
    vpd = es - ea
    
    return vpd

def update_vpd_data(data_xarray: xr.Dataset) -> xr.Dataset:
    """
    Updates the 'VPD' (Vapor Pressure Deficit) in the xarray dataset.
    
    Parameters:
    - data_xarray (xr.Dataset): The xarray dataset to update.
    - counts (Dict[str, int]): Dictionary containing the counts of missing values for different variables.
    
    Returns:
    - xr.Dataset: Updated xarray dataset with 'VPD' data.
    """
    
    # If missing, calculate VPD from RH and Tair
    vpd_array = calculate_vpd(
         data_xarray.RH.values.flatten(),
        data_xarray.Tair.values.flatten()
    )
    # Update 'VPD' in data_xarray with calculated values
    data_xarray['VPD'] = xr.DataArray(vpd_array.reshape(1, 1, -1), dims=['x', 'y', 'time'])
    # Set attributes to indicate the source of the data
    attributes_VPD = {'method': 'calculated from RH and Tair'}
    data_xarray['VPD'].attrs.update(attributes_VPD)
    
    return data_xarray

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


def get_co2_data(longitude, latitude, start_time, end_time, file_path, resampling_interval = '30min'):
    """
    Retrieve CAMS CO2 data for a specific location and time period.
    
    Parameters:
    longitude (float): Longitude of the location.
    latitude (float): Latitude of the location.
    start_time (str): Start time in the format 'YYYY-MM-DD'.
    end_time (str): End time in the format 'YYYY-MM-DD'.
    file_path (str): Path to the CAMS netCDF file.
    
    Returns:
    co2_data (np.ndarray): Array of CO2 data in parts per million (ppm). Since co2 data for cams is in kg/kg.
    """
     # Constants
    M_CO2 = 44.01  # Molar mass of CO2 in g/mol
    M_dry_air = 28.97  # Molar mass of dry air in g/mol
    
     # Conversion factor from kg/kg to ppm
    conversion_factor = 1e6  # ppm
    
    # Open the CAMS dataset
    cams = xr.open_dataset(file_path).sortby('time')
    
    # Select the nearest location to the provided longitude and latitude
    cams_location_selected = cams.sel(latitude=latitude, longitude=longitude, method='nearest')
    
    # Select the time range
    cams_date_selected = cams_location_selected.sel(time=slice(start_time, end_time))
    
    # Extract time and CO2 variables
    time_data = cams_date_selected['time'].values
    co2_data = cams_date_selected['co2'].values.reshape(-1)
    
    # Create a pandas DataFrame
    df = pd.DataFrame({'time': time_data, 'co2': co2_data})
    
    # Set 'time' column as index
    df.set_index('time', inplace=True)
    
    # Resample to 30-minute intervals and forward fill missing values
    df_filled = df.resample(resampling_interval).ffill()
    
    # Extend the index until 'end_time' and forward fill
    end_date_extend = pd.to_datetime(end_time)
    df_filled_new = df_filled.reindex(pd.date_range(start=df_filled.index.min(), end=end_date_extend, freq='30min')).ffill()
    
    # Extract CO2 data as numpy array
    co2 = np.array(df_filled_new['co2'])
    
    # Convert kg/kg to ppm
    co2_ppm = co2 * (conversion_factor * M_dry_air / M_CO2)
    
    return co2_ppm

def update_co2_data(data_xarray, cams_path, counts = None):
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
    if 'CO2air' not in data_xarray.data_vars:
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
    elif (isinstance(counts, dict) and counts.get('CO2air', 0) > 0):
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

def calculate_vapor_pressure_deficit_from_temperatures(
    air_temperatures: np.ndarray,
    dew_point_temperatures: np.ndarray,
    return_components: Optional[bool] = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Calculate the vapor pressure deficit (VPD) given air temperatures and dew point temperatures in degrees Celsius.
    
    Uses Magnus-Tetens relationship (Murray,1967) and returns results in kiloPascals (KPa).
    Optionally returns saturation vapor pressure (esat), actual vapor pressure (e), and VPD.
    
    Parameters:
    air_temperatures (np.ndarray): Air temperatures in degrees Celsius.
    dew_point_temperatures (np.ndarray): Dew point temperatures in degrees Celsius.
    return_components (Optional[bool]): If True, return esat, e, and VPD; otherwise, return only VPD.

    Returns:
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        If return_components is False: VPD in hPa.
        If return_components is True: Tuple containing (esat, e, VPD) all in hPa.
    """
    # Define constants
    constant_a = 6.11E-2  # Scientific notation for 6.11 * 10^-2 in hPa
    
    if isinstance(air_temperatures, np.ndarray) and isinstance(dew_point_temperatures, np.ndarray):
        constant_b = np.where(air_temperatures >= 0, 17.268, 21.874)
        constant_c = np.where(air_temperatures >= 0, 237.29, 265.49)
        
        # Calculate saturation vapor pressure (esat) in hPa
        esat = constant_a * np.exp((constant_b * air_temperatures) / (air_temperatures + constant_c))
        
        # Calculate actual vapor pressure (e) in hPa
        e = constant_a * np.exp((constant_b * dew_point_temperatures) / (dew_point_temperatures + constant_c))
        
        # Calculate vapor pressure deficit (VPD) in hPa
        vpd = esat - e
        
        if return_components:
            return esat, e, vpd
        else:
            return vpd
    else:
        raise TypeError("Unsupported type for air_temperatures and dew_point_temperatures. Must be numpy.ndarray.")

def calculate_wind_speed_from_u_v(u_component_wind: np.ndarray, v_component_wind: np.ndarray) -> np.ndarray:
    """
    Calculate the wind speed from the U and V components of the wind.

    Args:
        u_component_wind (np.ndarray): Array of the U component of the wind (zonal wind component).
        v_component_wind (np.ndarray): Array of the V component of the wind (meridional wind component).

    Returns:
        np.ndarray: Array of wind speeds calculated from the U and V components.
    """
    # Calculate wind speed using the Pythagorean theorem
    wind_speed = np.sqrt(u_component_wind**2 + v_component_wind**2)
    return wind_speed


def convert_accumulated_values_to_hourly_values(accumulated_array: np.ndarray) -> np.ndarray:
    """
    Convert accumulated values to hourly values by computing differences between consecutive hours.

    Args:
        accumulated_array (np.ndarray): 1D or 2D array of accumulated values.

    Returns:
        np.ndarray: 2D array of hourly values, where each row corresponds to a day.
    """
    # Reshape the array to have 24 columns (for hours) and as many rows as necessary
    # Ensure that the total number of elements is divisible by 24
    num_hours = accumulated_array.size
    if num_hours % 24 != 0:
        raise ValueError("The size of the array must be divisible by 24.")
    
    trial = accumulated_array.reshape(-1, 24)
    trial_copy = trial.copy()

    # Compute the difference between consecutive hours
    # trial_shifted = trial[:,1:] is not used in the final calculation, so it's removed
    diff = trial[:, 1:] - trial[:, :-1]
    trial_copy[:, 1:] = diff

    # For the first hour of each day, retain the original value
    trial_copy[:, 0] = trial[:, 0]  # This retains the first hour's value as it is

    return trial_copy.flatten() 


