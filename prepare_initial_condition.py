# This script processes initial condition data from ERA5 Land for ICOS Stations in the format pystemmus scope reads.

# Importing necessary libraries
import pandas as pd
import os
import xarray as xr

# Define paths
initial_condition_path = "/home/khanalp/code/PhD/preprocessICOSdata/csvs/03_ERA5Land_initial_condition.csv"
output_path = "/home/khanalp/data/processed/initialcondition_pystemmus"

# Read the initial condition data CSV file
df_intial_condition = pd.read_csv(initial_condition_path, index_col=[0])

# Find rows with missing values
rows_with_missing_values = df_intial_condition[df_intial_condition.isna().any(axis=1)]
station_missing_initial_condition = rows_with_missing_values.index.to_list()

# Convert the 'image_date' column to datetime format
df_intial_condition['image_date'] = pd.to_datetime(df_intial_condition['image_date'])

# Dictionary for variable renaming
rename = {
    'skin_temperature': 'skt',
    'soil_temperature_level_1': 'stl1',
    'soil_temperature_level_2': 'stl2',
    'soil_temperature_level_3': 'stl3',
    'soil_temperature_level_4': 'stl4',
    'volumetric_soil_water_layer_1': 'swvl1',
    'volumetric_soil_water_layer_2': 'swvl2',
    'volumetric_soil_water_layer_3': 'swvl3',
    'volumetric_soil_water_layer_4': 'swvl4'
}

# Iterate over rows to create xarray Datasets
for index, row in df_intial_condition.iterrows():
    # Skip processing if the output file for the station already exists
    if any(index in filename for filename in os.listdir(output_path)):
        print(f"Output file already exists for station {index}. Skipping...")
        continue

    # Convert the row to a dictionary
    row_dict = row.to_dict()  
    station_name = index

    # Check if the station has missing initial condition data
    if station_name not in station_missing_initial_condition:
        print(f"Processing {station_name}")
        
        # Extract latitude, longitude, and image date
        lat = row_dict.pop('latitude')  
        lon = row_dict.pop('longitude')  
        date = row_dict.pop('image_date')  

        # Create an xarray Dataset from the remaining dictionary
        xds = xr.Dataset(row_dict)

        # Add latitude, longitude, and image_date as coordinates
        xds.coords['latitude'] = lat
        xds.coords['longitude'] = lon
        xds.coords['time'] = date

        # Rename variables according to the defined mapping
        xds_renamed = xds.rename(rename)

        # Convert variable data types to float32
        for var_name in xds_renamed.data_vars:
            xds_renamed[var_name] = xds_renamed[var_name].astype('float32')

        # Convert coordinates to float32
        xds_renamed['latitude'] = xds_renamed['latitude'].astype('float32')
        xds_renamed['longitude'] = xds_renamed['longitude'].astype('float32')

        # Construct the filename
        filename = f"{station_name}_{date.date()}_InitialCondition.nc"

        # Save the xarray Dataset to NetCDF format
        xds_renamed.to_netcdf(os.path.join(output_path, filename))
        print(f"Processing {station_name} completed")
