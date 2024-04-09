"""
Author: [Prajwal Khanal]
Date: [2024 April 9 @ 9:39 AM @ Berfloplein 1A, Hengelo]
Purpose: Generate configuration files for STEMMUS_SCOPE based on ICOS data.
"""

import os
import pandas as pd
import xarray as xr
from datetime import datetime

# Define paths
path_input_pystemmus = "/home/khanalp/data/processed/input_pystemmus"
output_file_path = "/home/khanalp/code/PhD/preprocessICOSdata/output/csvs"
filepath_ICOS_config = "/home/khanalp/STEMMUSSCOPE/STEMMUS_SCOPE/ICOS_sites"

# List all files in the directory
files = os.listdir(path_input_pystemmus)

# Filter out only NetCDF files
nc_files = [file for file in files if file.endswith('.nc')]

# Initialize an empty list to store data
data = []

# Iterate over NetCDF files
for file in nc_files:
    path = os.path.join(path_input_pystemmus, file)
    station_name = file.split('_')[1]
    ds = xr.open_dataset(path)
    
    # Get the minimum and maximum time values
    min_time = ds.time.min().values
    max_time = ds.time.max().values
    
    # Format the time values
    min_time_str = datetime.utcfromtimestamp(min_time.tolist() / 1e9).strftime('%Y-%m-%dT%H:%M')
    max_time_str = datetime.utcfromtimestamp(max_time.tolist() / 1e9).strftime('%Y-%m-%dT%H:%M')
    
    # Append data to the list
    data.append([station_name, min_time_str, max_time_str])
    
    # Close the dataset
    ds.close()

# Create a DataFrame from the list of data
df = pd.DataFrame(data, columns=['Station_Name', 'Start_Time', 'End_Time'])

# Save DataFrame to a CSV file
df.to_csv(os.path.join(output_file_path, "info_for_configfile.csv"))

# Iterate over DataFrame rows to generate config files
for index, row in df.iterrows():
    station_name = row['Station_Name']
    start_time = row['Start_Time']
    end_time = row['End_Time']
    
    print(station_name,start_time,end_time)
    # Define config template
    config_template = f"""\
WorkDir=/home/khanalp/STEMMUSSCOPE/STEMMUS_SCOPE/ICOS_sites/{station_name}/
SoilPropertyPath=/home/khanalp/STEMMUSSCOPE/STEMMUS_SCOPE/input/SoilProperty/
ForcingPath=/home/khanalp/data/processed/input_pystemmus/
Location={station_name}
directional=/home/khanalp/STEMMUSSCOPE/STEMMUS_SCOPE/input/directional/
fluspect_parameters=/home/khanalp/STEMMUSSCOPE/STEMMUS_SCOPE/input/fluspect_parameters/
leafangles=/home/khanalp/STEMMUSSCOPE/STEMMUS_SCOPE/input/leafangles/
radiationdata=/home/khanalp/STEMMUSSCOPE/STEMMUS_SCOPE/input/radiationdata/
soil_spectrum=/home/khanalp/STEMMUSSCOPE/STEMMUS_SCOPE/input/soil_spectrum/
input_data=/home/khanalp/STEMMUSSCOPE/STEMMUS_SCOPE/input/input_data.xlsx
InitialConditionPath=/home/khanalp/data/processed/initialcondition_pystemmus/
StartTime={start_time}
EndTime={end_time}
InputPath=
OutputPath=
"""

    # Construct the directory path
    station_dir = os.path.join(filepath_ICOS_config, station_name)
    
    # Create station folder if it doesn't exist
    if not os.path.exists(station_dir):
        os.makedirs(station_dir)

    # Write config file
    with open(os.path.join(station_dir, "config_file.txt"), "w") as file:
        file.write(config_template)
