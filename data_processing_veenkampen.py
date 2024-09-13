import os
import re
import pandas as pd
from typing import List,Tuple,Dict,Union
import xarray as xr
import numpy as np

import re
from typing import List
from datetime import datetime

from utils import resample_and_aggregate,add_station_info_to_xarray,add_lai_data_to_xarray
import matplotlib.pyplot as plt


def get_files_for_year_range(
        files: List[str],
        start_year: int,
        end_year: int,
) -> List[str]:

    # Regular expression to extract the year and month from filenames (format: YYYY_MM)
    date_pattern = re.compile(r'(\d{4}_\d{2})')

    # List to store the filtered files that match the criteria
    filtered_files = []

    # Iterate over all files in the specified list
    for file in files:
        # Search for the date pattern in the filename
        match = date_pattern.search(file)

        # If a date is found in the filename
        if match:
            # Extract the year part from the matched date (format: YYYY_MM)
            file_year = int(match.group(0).split('_')[0])
            file_month = int(match.group(0).split('_')[1])

            # Check if the extracted year falls within the specified range
            if start_year <= file_year <= end_year:
                # Append file and extracted date for sorting later
                filtered_files.append((file, file_year, file_month))

    # Sort files based on the year and month in ascending order
    filtered_files.sort(key=lambda x: (x[1], x[2]))

    # Extract the filenames from the sorted list
    sorted_files = [file for file, _, _ in filtered_files]

    # Return the list of sorted files
    return sorted_files


def extract_time_and_data(file_path: str, variables: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    # Read the Timestamp column separately
    timestamps = pd.read_csv(file_path, usecols=["Timestamp"], skiprows=[1], squeeze=True).values

    # Read the rest of the data while skipping the first row
    df_data = pd.read_csv(file_path, skiprows=[1], usecols=variables)
    # Transpose to make each row a variable
    data_array = df_data[variables].to_numpy().T
    return  timestamps, data_array


def extract_variables_to_numpy(file_location: str, files: List[str], variables: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    data_list = []  # List to store data arrays for each file
    timestamps_list = []  # List to store Timestamp arrays for each file

    for file in files:
        file_path = os.path.join(file_location, file)
        
        # Extract units, data, and timestamps
        timestamps,data = extract_time_and_data(file_path, variables)

        # Append data and timestamps to their respective lists
        data_list.append(data)
        timestamps_list.append(timestamps)

    # Concatenate data horizontally along the time axis: shape will be (number_of_variables, total_time)
    combined_data = np.hstack(data_list)
    
    # Concatenate all timestamps: shape will be (total_time,)
    combined_timestamps = np.concatenate(timestamps_list)
    combined_timestamps = np.array(combined_timestamps, dtype='datetime64[ns]')
    return combined_timestamps,combined_data

def extract_units_from_csv(file_location: str, files: List[str], variables: List[str]) -> Tuple[Dict[str, str]]:
    units_dict={}
    file_path = os.path.join(file_location,files[0])
    df_units = pd.read_csv(file_path,nrows=1)
    for variable in variables:
            units_dict[variable] = df_units[variable].iloc[0]
    return units_dict

def create_xarray_dataset(
    combined_data: np.ndarray,
    combined_timestamps: np.ndarray,
    variables: List[str],
    x_coords: np.ndarray,
    y_coords: np.ndarray
) -> xr.Dataset:
    
    # Ensure combined_timestamps is a datetime64 array
    combined_timestamps = pd.to_datetime(combined_timestamps)
    
    # Create a dictionary of data variables
    data_dict = {variable: (("x", "y", "time"), combined_data[index, :].reshape(len(x_coords), len(y_coords), -1))
                 for index, variable in enumerate(variables)}

    # Create the xarray Dataset
    dataset = xr.Dataset(
        data_vars=data_dict,
        coords={
            "x": x_coords,  # Longitude or x coordinate
            "y": y_coords,  # Latitude or y coordinate
            "time": combined_timestamps  # Time coordinate
        }
    )

    return dataset

def set_variable_units(dataset: xr.Dataset, units_dict: Dict[str, str]) -> None:

    for variable, unit in units_dict.items():
        if variable in dataset.data_vars:  # Check if variable is in the dataset
            dataset[variable].attrs['units'] = unit
        else:
            print(f"Warning: Variable '{variable}' not found in dataset.")
    
def combine_datasets(ds_1min_resampled_30mins: xr.Dataset, dataset_30mins: xr.Dataset) -> xr.Dataset:

    # Merge the datasets
    combined_dataset = xr.merge([ds_1min_resampled_30mins, dataset_30mins])

    return combined_dataset

def count_nans(data_xarray: xr.Dataset) -> dict:
    
    nan_counts = {}

    for var_name in data_xarray.data_vars:
        # Count NaNs in each variable
        nan_count = data_xarray[var_name].isnull().sum().item()
        nan_counts[var_name] = nan_count

    return nan_counts

def find_nan_times(data_xarray: xr.Dataset) -> dict:

    nan_times = {}

    for var_name in data_xarray.data_vars:
        # Get the variable data
        data = data_xarray[var_name]
        
        # Check if the variable data is numeric
        if np.issubdtype(data.dtype, np.number):
            # Find NaN indices
            nan_indices = np.where(np.isnan(data))
            
            # Get the time dimension (assuming it is named 'time')
            time_coord = data_xarray.coords['time']
            
            # Extract times corresponding to NaN values
            nan_times[var_name] = time_coord.values[nan_indices[0]]
        else:
            # Skip non-numeric variables
            nan_times[var_name] = np.array([])

    return nan_times

def find_unique_nan_times(data_xarray: xr.Dataset) -> np.ndarray:
   
    nan_times_set = set()

    for var_name in data_xarray.data_vars:
        # Get the variable data
        data = data_xarray[var_name]
        
        # Check if the variable data is numeric
        if np.issubdtype(data.dtype, np.number):
            # Find NaN indices
            nan_indices = np.where(np.isnan(data))
            
            # Get the time dimension (assuming it is named 'time')
            time_coord = data_xarray.coords['time']
            
            # Extract times corresponding to NaN values
            nan_times_set.update(time_coord.values[nan_indices[0]])
    
    # Return unique sorted times
    return np.sort(np.array(list(nan_times_set)))

def find_missing_times(
    ds: np.ndarray,
    start_time: Union[str, pd.Timestamp], 
    end_time: Union[str, pd.Timestamp], 
    freq: str, 
) -> np.ndarray:
    """
    Find missing times in a dataset compared to a generated datetime range.
    
    Args:
        start_time (Union[str, pd.Timestamp]): Start date of the datetime range.
        end_time (Union[str, pd.Timestamp]): End date of the datetime range.
        freq (str): Frequency of the datetime range (e.g., '30T' for 30 minutes).
        ds (np.ndarray): Numpy array of datetime64 values representing the dataset times.
    
    Returns:
        np.ndarray: Numpy array of datetime64 values that are missing in the dataset.
    """
    # Generate the datetime range based on the provided parameters
    datetime_range = pd.date_range(start=start_time, end=end_time, freq=freq)
    
    # Convert to numpy array of datetime64[ns]
    datetime_array = datetime_range.to_numpy(dtype='datetime64[ns]')
    
    # Convert to pandas datetime for easier comparison
    datetime_array_pd = pd.to_datetime(datetime_array)
    ds_pd = pd.to_datetime(ds.time.values)
    
    # Find times in datetime_array that are missing in ds
    missing_times = datetime_array_pd[~datetime_array_pd.isin(ds_pd)]
    
    # Convert back to numpy array
    return missing_times.to_numpy()

def save_array_txt(array: np.ndarray, filename: str):
    with open(filename, 'w') as file:
        for dt in array:
            # Convert datetime64 to string
            file.write(f"{dt.astype(str)}\n")


    

if __name__ == "__main__":   
    
    # file paths 
    veenkampen_file_location = "/home/khanalp/data/fluxsites_NL/incoming/veenkampen"
    station_details = '/home/khanalp/code/PhD/preprocessICOSdata/csvs/02_station_with_elevation_heightcanopy.csv'  
    lai_modis_path = "/home/khanalp/data/processed/lai/modis"
    ERA5land_data_path = "/home/khanalp/code/PhD/preprocessICOSdata/csvs/ERA5_land_data_veenkampen.csv"
    resampled_dataset_30mins_path = os.path.join(veenkampen_file_location, "intermediate", "combined_dataset_30mins.nc")
    
    # read csv files 
    df_ERA5land = pd.read_csv(ERA5land_data_path)
    df_ERA5land['time'] = pd.to_datetime(df_ERA5land['Column1'])
    df_ERA5land.drop(columns=['Column1'], inplace=True)
    # Load station details from CSV
    df_station_details = pd.read_csv(station_details) 
    
    # Get lists of files filtered by interval type because they are downloaded separately
    
    files_30mins = [file for file in os.listdir(veenkampen_file_location) if "30mins" in file]
    files_1min = [file for file in os.listdir(veenkampen_file_location)
              if file not in files_30mins and "WTD" not in file]
    
    # THese are start and end_year
    start_year = 2015
    end_year = 2023
    
    # THese are variables available in 30 mins and 1 min resolution respsectively. 
    variables_30mins = ['air_temperature', 'VPD', 'air_pressure', 'wind_speed', 'co2_mole_fraction', 'RH', 'specific_humidity']
    variables_1min = ['SW_IN_1_1_1', 'LW_IN_1_1_1', 'P_1_1_7']
    
    # Define station name and grab details for it. 
    station_name = "NL-Vee"
    station_info = df_station_details[df_station_details['station_name'] == station_name]
    
    # Sample data: Assume data for 7 variables across 100 time points at 1 x 1 spatial grid
    x_coords = np.array(station_info['longitude'])  # Example x coordinate (e.g., longitude)
    y_coords = np.array(station_info['latitude'])  # Example y coordinate (e.g., latitude)

    # Filter files by year range for each interval type
    files_30mins_start_end = get_files_for_year_range(files_30mins, start_year, end_year)
    files_1min_start_end = get_files_for_year_range(files_1min, start_year,end_year)

    #get variables in array
    combined_timestamps_30mins,combined_data_30mins = extract_variables_to_numpy(veenkampen_file_location,files_30mins_start_end,variables_30mins)
    units_30mins = extract_units_from_csv(veenkampen_file_location,files_30mins_start_end,variables_30mins)
    
# Extract data to xarray 
    dataset_30mins = create_xarray_dataset(combined_data_30mins, combined_timestamps_30mins, variables_30mins, x_coords, y_coords)
    set_variable_units(dataset_30mins, units_30mins)
    
   
    # Process and save datasets if not already saved
    if not os.path.exists(resampled_dataset_30mins_path):
        combined_timestamps_1min, combined_data_1min = extract_variables_to_numpy(veenkampen_file_location,files_1min_start_end,variables_1min)
        units_1min = extract_units_from_csv(veenkampen_file_location,files_1min_start_end,variables_1min)
        dataset_1min = create_xarray_dataset(combined_data_1min, combined_timestamps_1min, variables_1min, x_coords, y_coords)
        set_variable_units(dataset_1min, units_1min)
        ds_1min_resampled_30mins = resample_and_aggregate(dataset=dataset_1min,freq='30min',sum_variable='P_1_1_7')
        ds_1min_resampled_30mins.to_netcdf(resampled_dataset_30mins_path)
    else: 
        ds_1min_resampled_30mins = xr.open_dataset(resampled_dataset_30mins_path)
    
    #Some time steps are completely missing in dataset_30mins which i want to fill with ERA5 land data. 
    missing_times_in_dataset_30mins = find_missing_times(ds=dataset_30mins,start_time="2015-01-01",end_time="2023-12-31",freq="30T")
    # save_array_txt(missing_times_in_dataset_30mins,filename="missing_times_veenkampen.txt")
    
    combined_dataset = dataset_30mins.combine_first(ds_1min_resampled_30mins) # this combine in a way the date time not present in first dataset will be missing
    
    for variable in combined_dataset.data_vars:
        print(variable,combined_dataset[variable].units)
    
    time_to_select = "2015-01-27T06:00:00.000000000"
    # Select the data for the specific time
    selected_data = combined_dataset.sel(time=time_to_select, method='nearest') 
    
    df = combined_dataset.to_dataframe()
    
    df.to_csv(os.path.join(veenkampen_file_location,"insitu_major_meteorological_variable.csv"))
    
    
     # Convert xarray Dataset to DataFrame
    ds_df = dataset_30mins.to_dataframe().reset_index()

    # Convert missing_times to DataFrame
    missing_times_df = pd.DataFrame(missing_times_in_dataset_30mins, columns=['time'])
    
    # Filter the external DataFrame `df` for only the missing times
    missing_data_df = df_ERA5land[df_ERA5land['time'].isin(missing_times_df['time'])]

    # Merge the original Dataset DataFrame with the missing data
    filled_df = pd.merge(ds_df, missing_data_df, on='time', how='outer')

    # Convert the merged DataFrame back to an xarray Dataset
    filled_ds = filled_df.set_index(['time']).to_xarray()
    
    
    
    
    # data_xarray = combine_datasets(dataset_30mins,ds_1min_resampled_30mins)
    # # Add station-specific information, MODIS LAI data, and update CO2 and VPD data
    # data_xarray = add_station_info_to_xarray(data_xarray, station_info)
    # # data_xarray = add_lai_data_to_xarray(data_xarray, lai_modis_path, station_name)
    # count_NA = count_nans(data_xarray=data_xarray)
    # nan_times = find_nan_times(data_xarray)
    
    
    
    
    
    # Find missing times
    
#     unique_missing_times = find_unique_nan_times(data_xarray)
    
#     # Concatenate both arrays
#     combined_array = np.concatenate((missing_times, unique_missing_times))
    
    
#     # Get unique datetime values and sort them
#     unique_array = np.unique(combined_array)
    
# def save_datetime_array_to_txt(datetime_array: np.ndarray, filename: str):
#     with open(filename, 'w') as file:
#         for dt in datetime_array:
#             # Convert datetime64 to string
#             file.write(f"{dt.astype(str)}\n")

# # Example usage
# save_datetime_array_to_txt(unique_array, 'missing_dates_veenkampen.txt')

# import numpy as np
# from datetime import datetime, timedelta

# def adjust_to_hourly(datetime_array: np.ndarray) -> np.ndarray:
#     # Convert NumPy datetime64 array to list of Python datetime objects
#     datetime_list = [pd.to_datetime(dt).to_pydatetime() for dt in datetime_array]
    
#     # Create a set to hold unique adjusted datetime objects
#     adjusted_set = set()
    
#     for dt in datetime_list:
#         if dt.minute == 30:
#             # Add both the preceding and following hours
#             adjusted_set.add(dt.replace(minute=0, second=0, microsecond=0))
#             adjusted_set.add(dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
#         else:
#             # Add the original datetime object
#             adjusted_set.add(dt.replace(minute=0, second=0, microsecond=0))
    
#     # Convert set back to NumPy array and sort it
#     adjusted_array = np.array(sorted(adjusted_set), dtype='datetime64')
    
#     return adjusted_array

# # Example usage
# # datetime_array = np.array(['2024-01-01T05:30'], dtype='datetime64')
# adjusted_array = adjust_to_hourly(unique_array)
# save_datetime_array_to_txt(adjusted_array, 'missing_dates_veenkampen_hourly.txt')


    
   



    


    

    
    
    
    
    
