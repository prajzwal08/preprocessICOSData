# Title: Processing Initial Condition Data from ERA5 Land for ICOS Stations in the format pystemmusscope reads.
# Author: Prajwal Khanal
# Date: Friday, 22 March 2024 @ 11:51 AM @ Boston Public Library

# Importing necessary libraries
import ee
import pandas as pd
import os
import xarray as xr
from datetime import datetime


#Paths
initial_condition_path = "/home/khanalp/code/PhD/preprocessICOSdata/output/csvs/ERA5LandInitialcondition.csv"
output_path = "/home/khanalp/data/processed/initialcondition_pystemmus"

# Reading the initial condition data CSV file
df_intial_condition = pd.read_csv(initial_condition_path, index_col=[0])

# Finding rows with missing values
rows_with_missing_values = df_intial_condition[df_intial_condition.isna().any(axis=1)]
station_missing_initial_condition = rows_with_missing_values.index.to_list()


# Converting the 'image_date' column to datetime format
df_intial_condition['image_date'] = pd.to_datetime(df_intial_condition['image_date'])

# Dictionary for variable renaming
rename = {'skin_temperature': 'skt',
          'soil_temperature_level_1': 'stl1',
          'soil_temperature_level_2': 'stl2',
          'soil_temperature_level_3': 'stl3',
          'soil_temperature_level_4': 'stl4',
          'volumetric_soil_water_layer_1': 'swvl1',
          'volumetric_soil_water_layer_2': 'swvl2',
          'volumetric_soil_water_layer_3': 'swvl3',
          'volumetric_soil_water_layer_4': 'swvl4'}

# Iterating over rows to create xarray Datasets
for index, row in df_intial_condition.iterrows():
    row_dict = row.to_dict()  # Convert the row to a dictionary
    station_name = index
    if station_name not in station_missing_initial_condition:
        print(f"Processing {station_name}")
        lat = row_dict.pop('latitude')  # Extracting latitude
        lon = row_dict.pop('longitude')  # Extracting longitude
        date = row_dict.pop('image_date')  # Extracting image date

        # Creating an xarray Dataset from the dictionary
        xds = xr.Dataset(row_dict)

        # Adding latitude, longitude, and image_date as coordinates
        xds.coords['latitude'] = lat
        xds.coords['longitude'] = lon
        xds.coords['time'] = date

        # Renaming variables
        xds_renamed = xds.rename(rename)

        # Converting variable data type to float32
        for var_name in xds_renamed.data_vars:
            xds_renamed[var_name] = xds_renamed[var_name].astype('float32')

        # Converting coordinates to float32
        xds_renamed['latitude'] = xds_renamed['latitude'].astype('float32')
        xds_renamed['longitude'] = xds_renamed['longitude'].astype('float32')

        # Constructing the filename
        filename = f"{station_name}_{date.date()}_InitialCondition.nc"

        # Saving the xarray Dataset to NetCDF format
        xds_renamed.to_netcdf(os.path.join(output_path, filename))
        print(f"Processing {station_name} completed")
        #break
