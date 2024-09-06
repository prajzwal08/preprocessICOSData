# Title: Retrieving ERA5 Land Initial Condition Data for ICOS Stations using Google Earth Engine
# Author: [Prajwal Khanal]
# Date: [22 March 2024 @ 11:34 AM @ Boston Public Library]

# Importing necessary libraries
import ee
import pandas as pd
import os

# Authenticating and initializing Earth Engine
ee.Authenticate()
ee.Initialize()

# Path variables
station_details = '/home/khanalp/code/PhD/preprocessICOSdata/csvs/02_station_with_elevation_heightcanopy.csv'
input_pystemmus_location = "/home/khanalp/data/processed/input_pystemmus"
output_path = "/home/khanalp/code/PhD/preprocessICOSdata/csvs"
output_filename = "03_ERA5Land_initial_condition.csv"

# List all files in the directory
files_in_directory = os.listdir(input_pystemmus_location)
nc_files = [file for file in files_in_directory if file.endswith(".nc")]

# Initializing a dictionary to store station information
station = {}

# Extracting station information from NetCDF files
for nc_file in nc_files:
    station_name = nc_file.split("_")[1]
    start_year = int(nc_file.split("_")[-1][:4])
    end_year = int(nc_file.split("_")[-1][5:9])
    station[station_name] = {'start_year': start_year, 'end_year': end_year}

# Reading station details from a CSV file
# Load station details from CSV
df_station_details = pd.read_csv(station_details)
df_output = pd.read_csv(os.path.join(output_path,output_filename),index_col=0)

# Iterate through each station in the station dictionary
for station_name, info in station.items():
    if station_name in df_station_details['station_name'].values:
        lat = df_station_details.loc[df_station_details['station_name'] == station_name, 'latitude'].values[0]
        lon = df_station_details.loc[df_station_details['station_name'] == station_name, 'longitude'].values[0]
        station[station_name]['latitude'] = lat
        station[station_name]['longitude'] = lon

# Required bands for retrieval
band_required = ['skin_temperature', 'soil_temperature_level_1', 'soil_temperature_level_2', 'soil_temperature_level_3', 'soil_temperature_level_4', 'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4']

# Empty list to store DataFrames for each station
dfs = []

# Loop through each station in the station dictionary
for station_name, station_info in station.items():
    if station_name not in df_output.index:
        print(station_name)
        latitude = station_info['latitude']
        longitude = station_info['longitude']
        start_year = station_info['start_year']
        end_year = station_info['end_year']
        point = ee.Geometry.Point(longitude, latitude)
        
        # Filter image collection for the specified year
        filtered_collection = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterDate(str(start_year) + "-01-01", str(start_year) + "-12-31")
        first_image = ee.Image(filtered_collection.first())
        image_date = ee.Date(first_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        values_dict = first_image.reduceRegion(reducer=ee.Reducer.first(), geometry=point, scale=10000).getInfo()
        
        # Create a DataFrame from the dictionary of band values
        station_df = pd.DataFrame(values_dict, index=[station_name])
        station_df = station_df[band_required]

        # Add latitude, longitude, and image date as columns
        station_df['latitude'] = latitude
        station_df['longitude'] = longitude
        station_df['image_date'] = image_date
        
        # Append the station DataFrame to the list
        dfs.append(station_df)

df_concated = pd.concat(dfs)

# Concatenate all DataFrames into a single DataFrame
if not df_concated.empty:
    df_output = pd.concat([df_output, df_concated])
    print("Initial conditions for some stations updated")
    df_output.to_csv(os.path.join(output_path,output_filename), index=True)
else: 
    print("Initial conditions for all stations are already available")

