# Author: [Prajwal Khanal]
# Date: [Thursday, March 21, 2024 @ Boston Public Library]
# Description: For ICOS (Integrated Carbon Observation System) data for specific stations,
# It calculates the percentage of missing data for each variable aggregated by unique year-month combinations 
# and saves the results to CSV files.

# Import necessary libraries
import pandas as pd
import xarray as xr
import numpy as np
import os

# Import utility functions from a custom module
from utils import list_folders_with_prefix, list_csv_files_in_folder, read_csv_file_with_station_name, select_rename_convert_to_xarray

# Define the paths for input and output data
ICOS_location = "/home/khanalp/data/ICOS2020"  # Path to the ICOS data
output_location = "/home/khanalp/code/PhD/preprocessICOSdata/output"  # Path to save the processed data

# Define the prefix used for folder filtering
prefix = "FLX"

def calculate_nan_percentage(data_array):
    """
    Calculate the percentage of missing data for each variable in an xarray dataset 
    aggregated by unique year-month combinations.

    Parameters:
    - data_array (xarray.Dataset): An xarray dataset containing the data variables along with their timestamps.

    Returns:
    - percentage_missing_data (pandas.DataFrame): A DataFrame containing the percentage of missing data for each 
      variable in each unique year-month combination.
    """
    # Calculate the number of NaN values in each unique year-month 
    data_dict = {'date': data_array.time}   
    # Loop through each variable in the xarray dataset
    for var_name, var_data in data_array.items():
        # Flatten the variable's data and add it to the dictionary
        data_dict[var_name] = pd.Series(var_data.values.flatten())

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data_dict)
    df['date'] = pd.to_datetime(df['date'])
    
    # Group by year and month
    grouped = df.groupby([df['date'].dt.year, df['date'].dt.month])
    group_size = grouped.size()  
    group_size.name = 'count'  
    missing_data_counts = grouped.apply(lambda x: x.isna().sum())
    merged = group_size.to_frame().join(missing_data_counts, how='outer')
    # Calculate the percentage of missing data for each column in each group
    percentage_missing_data = (merged.div(merged['count'], axis=0) * 100).drop(['count','date'], axis=1)
    return percentage_missing_data

# List folders and CSV files
folders = list_folders_with_prefix(ICOS_location, prefix)
csv_files = [os.path.join(ICOS_location, folder, file) for folder in folders for file in list_csv_files_in_folder(os.path.join(ICOS_location, folder), "FULLSET_HH")]

# Read station details
station_all = pd.read_csv("/home/khanalp/output/csvs/stationdetails.csv").set_index('station_name')

for index, station_info in station_all.iterrows():
    data_frame = read_csv_file_with_station_name(index, csv_files)
    if data_frame is None:
        print(f"CSV file not found for station {index}. Skipping...")
        continue
    #selecting required variables from ICOS data for input.
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

    #Renaming them 
    rename_mapping = {
        'TA_F':'Tair',
        'TA_F_QC':'Tair_qc',
        'SW_IN_F':'SWdown',
        'SW_IN_F_QC':'SWdown_qc',
        'LW_IN_F':'LWdown',
        'LW_IN_F_QC':'LWdown_qc',
        'VPD_F':'VPD',
        'VPD_F_QC':'VPD_qc',
        'PA_F':'Psurf',
        'PA_F_QC':'Psurf_qc',
        'P_F' : 'Precip',
        'P_F_QC':'Precip_qc',
        'WS_F':'Wind',
        'WS_F_QC':'Wind_qc',
        'CO2_F_MDS':'CO2air',
        'CO2_F_MDS_QC':'CO2air_qc'
        }

    #selected_variables = ['TA_F','SW_IN_F','LW_IN_F','VPD_F','PA_F','P_F','WS_F','CO2_F_MDS']
    # Get xarray with selected variables and after renaming .
    data_xarray = select_rename_convert_to_xarray(data_frame,selected_variables,rename_mapping)
    data_xarray = data_xarray.where(data_xarray != -9999, np.nan)
    
    # Example usage:
    nan_percentages_df = calculate_nan_percentage(data_xarray[['Tair','SWdown','LWdown','VPD','Psurf','Precip','Wind','CO2air']])
    #nan_percentages_df = nan_percentages_df.rename(columns={'date': 'count'})
    #saving 
    file_path = os.path.join(output_location, "csvs", "countNAs",f"{index}.csv")
    nan_percentages_df.to_csv(file_path)
    
    
