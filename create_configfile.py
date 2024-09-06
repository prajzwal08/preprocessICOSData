"""
Purpose: Generate configuration files for pystemmusscope
"""

import os
import pandas as pd
import xarray as xr

# Define paths
path_input_pystemmus = "/home/khanalp/data/processed/input_pystemmus"
output_file_path = "/home/khanalp/code/PhD/preprocessICOSdata/csvs"
filepath_ICOS_config = "/home/khanalp/code/PhD/preprocessICOSdata/configfiles"

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
    min_time_str = pd.to_datetime(min_time, unit='s').strftime('%Y-%m-%dT%H:%M')
    max_time_str = pd.to_datetime(max_time, unit='s').strftime('%Y-%m-%dT%H:%M')
    
    # Append data to the list
    data.append([station_name, min_time_str, max_time_str])
    
    # Close the dataset
    ds.close()

# Create a DataFrame from the list of data
df = pd.DataFrame(data, columns=['Station_Name', 'Start_Time', 'End_Time'])

# Save DataFrame to a CSV file
df.to_csv(os.path.join(output_file_path, "04_info_for_configfile.csv"))

# Iterate over DataFrame rows to generate config files
for index, row in df.iterrows():
    station_name = row['Station_Name']
    start_time = row['Start_Time']
    end_time = row['End_Time']
    
    # Define the directory path
    station_dir = os.path.join(filepath_ICOS_config, station_name)
    
    # Check if the directory already exists
    if not os.path.exists(station_dir):
        print(station_name)
        # Create station folder
        os.makedirs(station_dir)
        
        # Define config template
        config_template = f"""\
WorkDir=/home/khanalp/STEMMUS_SCOPE_model/STEMMUS_SCOPE_old/STEMMUS_SCOPE/ICOS_sites/{station_name}/
SoilPropertyPath=/home/khanalp/STEMMUS_SCOPE_model/STEMMUS_SCOPE_old/STEMMUS_SCOPE/input/SoilProperty/
ForcingPath=/home/khanalp/data/processed/input_pystemmus/
Location={station_name}
directional=/home/khanalp/STEMMUS_SCOPE_model/STEMMUS_SCOPE_old/STEMMUS_SCOPE/input/directional/
fluspect_parameters=/home/khanalp/STEMMUS_SCOPE_model/STEMMUS_SCOPE_old/STEMMUS_SCOPE/input/fluspect_parameters/
leafangles=/home/khanalp/STEMMUS_SCOPE_model/STEMMUS_SCOPE_old/STEMMUS_SCOPE/input/leafangles/
radiationdata=/home/khanalp/STEMMUS_SCOPE_model/STEMMUS_SCOPE_old/STEMMUS_SCOPE/input/radiationdata/
soil_spectrum=/home/khanalp/STEMMUS_SCOPE_model/STEMMUS_SCOPE_old/STEMMUS_SCOPE/input/soil_spectrum/
input_data=/home/khanalp/STEMMUS_SCOPE_model/STEMMUS_SCOPE_old/STEMMUS_SCOPE/input/input_data.xlsx
InitialConditionPath=/home/khanalp/data/processed/initialcondition_pystemmus/
StartTime={start_time}
EndTime={end_time}
InputPath=
OutputPath=
"""

        # Write config file
        with open(os.path.join(station_dir, "config_file.txt"), "w") as file:
            file.write(config_template)
    else:
        print(f"Configuration directory for {station_name} already exists. Skipping creation.")
