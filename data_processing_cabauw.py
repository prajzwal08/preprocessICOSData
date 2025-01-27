# Compare model output
# The data from 2003 to 2020 is from CESAR dataset. However, after 2020, data (currently) are available only on summer months. 
# Hence, I fill the forcing data from 2021 to 2023 from KNMI Station Cabauw, except for Psurf and LWdown which are filled from 2019 data.
import os
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List, Union
from utils.utils import get_lai_data,calculate_vapor_pressure, calculate_specific_humidity_mixing_ratio
from utils.unit_conversion import kilopascal_to_hectoPascal, kelvin_to_celsius, pascal_to_kilopascal, celsius_to_kelvin,hectopascal_to_pascal

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

def resample_and_aggregate(dataset: xr.Dataset, freq: str = '30min', sum_variable: str = '') -> xr.Dataset:
    """
    Resample the dataset to the specified frequency and apply aggregation functions.

    Parameters:
        dataset (xr.Dataset): The xarray dataset to resample.
        freq (str): Frequency for resampling (e.g., '30min').
        sum_variable (str): Variable to apply sum aggregation.

    Returns:
        xr.Dataset: Resampled and aggregated dataset.
    """
    # Check if sum_variable is specified
    if not sum_variable:
        mean_vars = [var for var in dataset.data_vars]
        # Apply mean aggregation to all variables
        ds_mean = dataset[mean_vars].resample(time=freq).mean(skipna=True)
        return ds_mean
    
    if sum_variable not in dataset.data_vars:
        raise ValueError(f"Variable '{sum_variable}' not found in the dataset.")
    
    mean_vars = [var for var in dataset.data_vars if var != sum_variable]
    sum_vars = [sum_variable]
    
    # Apply mean aggregation to all variables except the sum_variable
    ds_mean = dataset[mean_vars].resample(time=freq).mean(skipna=True)
    
    # Apply sum aggregation only to the specified variable
    ds_sum = dataset[sum_vars].resample(time=freq).sum(skipna=True)
    
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
    lai_modis_path = "/home/khanalp/data/processed/lai/modis/v1"
    output_directory = '/home/khanalp/data/processed/input_pystemmus/v1'
    station_details = '/home/khanalp/code/PhD/preprocessICOSdata/csvs/02_station_with_elevation_heightcanopy.csv'
    mauna_loa_path = '/home/khanalp/data/maunaloa_co2.csv'
    # Define the output file path
    combined_dataset_30mins_path = os.path.join(cabauw_file_location, "intermediate", "combined_dataset_30mins_v1.nc")
    resampled_KNMI_data_path = os.path.join(cabauw_file_location, "intermediate", "df_resampled.csv") # resampled 30 mins data covering period 2021 to 2023. 
    
    #Grab station details for Cabauw
    df_station_details = pd.read_csv(station_details)
    station_name = "NL-Cab"
    station_info = df_station_details[df_station_details['station_name'] == station_name]
    
    # read Mauna-Loa data 
    df_mauna_loa = pd.read_csv(mauna_loa_path,header = 0)
    df_mauna_loa['Date'] = pd.to_datetime(df_mauna_loa[['year', 'month', 'day']])
    df_mauna_loa = df_mauna_loa.set_index('Date')
    
    # Read resampled KNMI data
    df_data_2021_2023 = pd.read_csv(resampled_KNMI_data_path, index_col=0, parse_dates=True)
        
    # Process and save datasets if not already saved
    if not os.path.exists(combined_dataset_30mins_path):
        meteorological_file_keyword = "cesar_surface_meteo"
        radiation_file_keyword = "cesar_surface_radiation"
        flux_file_keyword = "cesar_surface_flux_lb1"
        combined_dataset_meteorology = get_combined_meteorological_dataset(cabauw_file_location, meteorological_file_keyword)
        combined_dataset_radiation = get_combined_meteorological_dataset(cabauw_file_location, radiation_file_keyword)
        combined_dataset_flux = get_combined_meteorological_dataset(cabauw_file_location, flux_file_keyword)
        # Load station details from CSV
        
        # Print long_name attributes for radiation variables (whats the purpose of this??)
        # for var_name in combined_dataset_flux.data_vars:
        #     var = combined_dataset_flux[var_name]
        #     if 'long_name' in var.attrs:
        #         print(f"{var_name}: {var.attrs['long_name']}")
        #     else:
        #         print(f"{var_name}: 'long_name' attribute not found")
        
       
        # Select and combine variables from meteorological and radiation datasets
        variables_from_meteorology = ['TA002', 'P0', 'RAIN', 'F010', 'RH002', 'Q002']
        variables_from_radiation = ['SWD', 'LWD']
        variables_from_flux = ['FCED','FG0','HSON','LEED','QNBAL']
        
        # variable info for flux
        variable_info = {
        'Rnet': {'rename_from': 'QNBAL', 'unit': 'W/m²', 'description': 'Net radiation from radiation balance SWD-SWU+LWD-LWU', 'missing_value': -9999},
        'Qh' :{'rename_from': 'HSON', 'unit': 'W/m', 'description': 'Sensible heat flux from sonic temperature corrected 3/5 m', 'missing_value': -9999},
        'Qg': {'rename_from': 'FG0', 'unit': 'W/m²', 'description': 'Soil heat flux at 0 cm, Fourrier extrapolation G05,G10', 'missing_value': -9999},
        'Qle': {'rename_from': 'LEED', 'unit': 'W/m²', 'description': 'Latent heat flux corrected 3/5 m', 'missing_value': -9999},
        'NEE': {'rename_from': 'FCED', 'unit': 'mg m-2 s-1', 'description': 'CO2 density flux Webb corrected', 'missing_value': -9999},
        }
        
        rename_mapping = {var_info['rename_from']: var_name for var_name, var_info in variable_info.items() if 'rename_from' in var_info}

        dataset_meteorology_selected = select_variables(combined_dataset_meteorology, variables_from_meteorology)
        dataset_radiation_selected = select_variables(combined_dataset_radiation, variables_from_radiation)
        dataset_flux_selected  = select_variables(combined_dataset_flux, variables_from_flux)
        combined_dataset = combine_datasets(dataset_meteorology_selected, dataset_radiation_selected)
        
        # Resample and aggregate dataset
        combined_dataset_30mins = resample_and_aggregate(combined_dataset, freq='30min', sum_variable='RAIN')
        dataset_flux_30mins = dataset_flux_selected.resample(time='30min').mean(skipna=True)
        dataset_flux_30mins = dataset_flux_30mins.rename(rename_mapping)
        dataset_flux_30mins.to_netcdf('/home/khanalp/data/processed/insituflux_with_qc/fluxes_NE-Cab_2000_2021_06.nc')
        combined_dataset_30mins.to_netcdf(combined_dataset_30mins_path)
    
    # Define alternative start and end dates
    # alternative_start_date = np.datetime64('2003-01-01 00:00:00')
    # alternative_end_date = np.datetime64('2023-12-31 00:00:00')

    ds_data = xr.open_dataset(combined_dataset_30mins_path)
    # Filter dataset based on available time range
    # data_xarray = combined_dataset_30mins.sel(time=slice(
    #     alternative_start_date if combined_dataset_30mins.time.values.min() < alternative_start_date else None,
    #     alternative_end_date if combined_dataset_30mins.time.values.max() > alternative_end_date else None
    # ))
   
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
    
    ds_renamed = ds_data.rename(rename_mapping)
    
    # Select only until 2020 because a lot data after 2020 is missing, which I will fill from the KNMI station "Cabauw"
    ds_2003_2020 = ds_renamed.sel(time  = slice(None, "2020-12-31 23:30:00"))

    # Expand dimensions to include 'x' and 'y' coordinates with dummy values
    ds_2003_2020 = ds_2003_2020.expand_dims({'x': [1], 'y': [2]})
    ds_2003_2020['x'] = ds_2003_2020['x'].astype('float64')
    ds_2003_2020['y'] = ds_2003_2020['y'].astype('float64')

    # Clip RH to 100 incase they are greater than that. 
    ds_2003_2020['RH'] = ds_2003_2020['RH'].clip(min=0, max=100) # Because some data are not within range
    
    # Calculate saturation vapor pressure (esat) for the period 2003 to 2020
    saturation_vapor_pressure_2003_2020 = kilopascal_to_hectoPascal(calculate_vapor_pressure(kelvin_to_celsius(ds_2003_2020.Tair.values.flatten()))) # esat as function of air temperature b
    actual_vapor_pressure_2003_2020 = ds_2003_2020['RH'].values.flatten() / 100 * saturation_vapor_pressure_2003_2020 # actual vapor pressure from RH
    
    ds_2003_2020['VPD'] = xr.DataArray((saturation_vapor_pressure_2003_2020 - actual_vapor_pressure_2003_2020).
                                       reshape(1, 1, -1), dims=['x', 'y', 'time'])
    # Set attributes to indicate the source of the data
    attributes_VPD = {'method': 'calculated from RH and Tair'}
    ds_2003_2020['VPD'].attrs.update(attributes_VPD)

    # Processing of data until 2020 is performed. 
    
    ################## Processing from 2021 ###################
    # get KNMI data in xarray
    ds_KNMI = xr.Dataset.from_dataframe(df_data_2021_2023).rename({'Rhumidity' : 'RH'})
    ds_KNMI = ds_KNMI.expand_dims({'x': [1], 'y': [2]})
    ds_KNMI['x'] = ds_KNMI['x'].astype('float64')
    ds_KNMI['y'] = ds_KNMI['y'].astype('float64')
    # Convert datetime coordinate of ds_KNMI to time
    ds_KNMI = ds_KNMI.rename({'datetime': 'time'})
    
    # the end_date is 2023
    end_date = pd.to_datetime('2023-12-31 23:30:00')

    # Select data from 2021 to 2023
    ds_2021_2023  = ds_KNMI.sel(time=slice(None,end_date))
    
    # Calculate VPD from Tair and Tdew
    saturation_vapor_pressure_2021_2023 = kilopascal_to_hectoPascal(calculate_vapor_pressure(ds_2021_2023['Tair'].values.flatten()))
    actual_vapor_pressure_2021_2023 = kilopascal_to_hectoPascal(calculate_vapor_pressure(ds_2021_2023['Tdew'].values.flatten()))
    
    ds_2021_2023['RH'] = xr.DataArray((actual_vapor_pressure_2021_2023 / saturation_vapor_pressure_2021_2023 * 100).
                                      reshape(1, 1, -1), dims=['x', 'y', 'time'])
    ds_2021_2023['VPD'] = xr.DataArray((saturation_vapor_pressure_2021_2023 - actual_vapor_pressure_2021_2023).
                                       reshape(1, 1, -1), dims=['x', 'y', 'time'])
    
    # For Air pressure (Psurf) from the 2021 to 2023, i want to fill the data from 2019. THis is because KNMI data seems problematic. 
    # Repeat values from 2019 three times to fill ds_KNMI_selected['Psurf']
    psurf_2019 = ds_renamed['Psurf'].sel(time=slice("2019-01-01 00:00:00", "2019-12-31 23:30:00")).values
    psurf_repeated = np.tile(psurf_2019, 3)
    ds_2021_2023['Psurf'] = xr.DataArray(psurf_repeated.reshape(1, 1, -1), dims=['x', 'y', 'time'])
    
    # Same for LWdown variable 
    lwdown_2019 = ds_renamed['LWdown'].sel(time=slice("2019-01-01 00:00:00", "2019-12-31 23:30:00")).values
    lwdown_repeated = np.tile(lwdown_2019, 3)
    ds_2021_2023['LWdown'] = xr.DataArray(lwdown_repeated.reshape(1, 1, -1), dims=['x', 'y', 'time'])
    
    # Calculate specific humidity and mixing ratio from actual vapor pressure and air pressure. Note: Presusure in kPa, ea in hPa
    specific_humidity_2021_2023, mixing_ratio_2021_2023 = calculate_specific_humidity_mixing_ratio(ea=actual_vapor_pressure_2021_2023, 
                                                                                                   pressure = pascal_to_kilopascal(hectopascal_to_pascal(ds_2021_2023['Psurf'].values.flatten())))
                                         
    # Qair from 2021 to 2023
    ds_2021_2023['Qair'] = xr.DataArray(specific_humidity_2021_2023.reshape(1, 1, -1),dims = ['x', 'y', 'time'])
    
    # Convert temperature in celsius to kelvin (for only data from 2021)
    ds_2021_2023['Tair'] = xr.DataArray(celsius_to_kelvin(ds_2021_2023['Tair'].values).reshape(1, 1, -1), dims=['x', 'y', 'time'])

    # Ensure that negative precipitation values are set to zero
    ds_2021_2023['Precip'] = ds_2021_2023['Precip'].clip(min=0) # See KNMI raw data file:  precipitation amount (in 0.1 mm) (-1 for <0.05 mm)

    # Combine the two datasets
    ds_combined = xr.concat([ds_2003_2020, ds_2021_2023], dim='time')

    # Sort the combined dataset by time
    ds_combined = ds_combined.sortby('time').drop_vars('Tdew')
    # Ensure that negative precipitation values are set to zero
   
    ## Add station specific information
    ds_combined['latitude'] = xr.DataArray(np.array(station_info['latitude']).reshape(1,-1), dims=['x','y'])
    ds_combined['longitude'] = xr.DataArray(np.array(station_info['longitude']).reshape(1,-1), dims=['x','y'])
    ds_combined['reference_height'] = xr.DataArray(np.array(station_info['measurement_height']).reshape(1,-1), dims=['x','y'])
    ds_combined['canopy_height'] = xr.DataArray(np.array(station_info['height_canopy_field_information']).reshape(1,-1), dims=['x','y'])
    ds_combined['elevation'] = xr.DataArray(np.array(station_info['elevation']).reshape(1,-1), dims=['x','y'])
    ds_combined['IGBP_veg_short'] = xr.DataArray(np.array(station_info['IGBP_short_name'], dtype = 'S200').reshape(1,-1),dims = ['x','y'])
    ds_combined['IGBP_veg_long'] = xr.DataArray(np.array(station_info['IGBP_long_name'], dtype = 'S200').reshape(1,-1),dims = ['x','y'])
    
    
    common_start_date,common_end_date,lai = get_lai_data(station_name=station_name,
                                           start_date=ds_combined.time.values.min(),
                                           end_date=ds_combined.time.values.max(),
                                           lai_modis_path=lai_modis_path)
    
    # Because LAI data might not available for the entire period, we need to select the common period for all datasets.
    ds_combined = ds_combined.sel(time=slice(common_start_date, common_end_date))
    # Check if the lengths match
    if len(lai) != len(ds_combined['time']):
        raise ValueError("Length mismatch between LAI data and xarray time dimension")
    
    # Add LAI data to data_xarray
    ds_combined['LAI'] = xr.DataArray(lai.reshape(1, 1, -1), dims=['x', 'y', 'time'])
    ds_combined['LAI_alternative'] = xr.DataArray(lai.reshape(1, 1, -1), dims=['x', 'y', 'time'])
    # data_xarray = add_lai_data_to_xarray(data_xarray, lai_modis_path, station_name)
    
    ## Fillinf Mauna LOa CO2 instead of CAMS because CAMS ends at 2020. While data for Cababuw is available until 2023.
    # For now, fill it with Mauna Loa data. 
    df_mauna_loa_selected = df_mauna_loa.loc[common_start_date:common_end_date]
    # Resample Mauna Loa CO2 data to 30-minute intervals
    df_mauna_loa_resampled = df_mauna_loa_selected.resample('30T').interpolate(method='linear')
    # Extend the resampled CO2 data to the common_end_date
    extended_index = pd.date_range(start=df_mauna_loa_resampled.index.min(), end=common_end_date, freq='30T')
    df_mauna_loa_extended = df_mauna_loa_resampled.reindex(extended_index).interpolate(method='linear')

    # Add extended CO2 data to the dataset
    ds_combined['CO2air'] = xr.DataArray(df_mauna_loa_extended['co2(ppm)'].values.reshape(1, 1, -1), dims=['x', 'y', 'time'])
    

    ## Changes are to be made here. 
    # time,co2_ppm = get_co2_data_from_cams(longitude  = data_xarray.longitude.values.flatten(), 
    #                                      latitude = data_xarray.latitude.values.flatten(), 
    #                                      start_time = data_xarray.time.values.min(), 
    #                                      end_time = data_xarray.time.values.max(), 
    #                                      file_path = cams_path, 
    #                                      resampling_interval = '30min') # Careful this will fill the data until 2023, cams data ends at 2020. 
 
    ## Unit in the format requied by the PyStemmus model
    
     # Convert precipitation from mm/30 min to mm/s
    ds_combined['Precip'] = ds_combined['Precip'] / (30 * 60)
    # Convert surface pressure from hPa to Pa
    ds_combined['Psurf'] = xr.DataArray(hectopascal_to_pascal(ds_combined['Psurf'].values.flatten()).reshape(1, 1, -1), dims=['x', 'y', 'time'])
    
    #Data format conversion
    for var_name in ds_combined.data_vars:
        if var_name not in ['IGBP_veg_short', 'IGBP_veg_long']:
            ds_combined[var_name] = ds_combined[var_name].astype('float32')

    # Fill missing values if any
    missing_vars = find_vars_with_missing_values(ds_combined)
    if missing_vars:
        for variable in missing_vars:
            ds_combined = fill_missing_values(ds_combined, variable)

    # Extract start and end years and construct the filename for saving
    time_pd = pd.to_datetime(ds_combined.time.values)
    start_year = time_pd.min().year
    end_year = time_pd.max().year
    filename = f"FLX_{station_name}_FLUXNET2015_FULLSET_{start_year}-{end_year}.nc"
    output_path = os.path.join(output_directory, filename)

    # Save the xarray dataset to NetCDF file
    ds_combined.to_netcdf(output_path)
    print("Dataset saved successfully.")
