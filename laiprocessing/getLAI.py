# Author: Prajwal Khanal
# Date: 10 April 2024 @ 9:12 PM @ Hengelo
# Purpose : To get smooth LAI, resampled at 30 mins after some processing into csv files. 

# Import necessary libraries
import os
import pandas as pd

# Importing functions to retrieve MODIS and Copernicus LAI data for a given station
from preprocessmodisLAI import get_modisLAI_for_station
from preprocesscopernicuslai import get_copernicusLAI_for_station

# Define file paths for station information and processed LAI data
file_path_station_info = "/home/khanalp/code/PhD/preprocessICOSdata/output/csvs/stationdetails.csv"
path_for_processed_modis_lai = "/home/khanalp/data/processed/lai/modis"
path_for_processed_copernicus_lai = "/home/khanalp/data/processed/lai/copernicus"

# Read station information from a CSV file into a pandas DataFrame
station_info = pd.read_csv(file_path_station_info)

# Print header/title
print("Processing MODIS and Copernicus LAI Data")

# Iterate over each row in the station information DataFrame
for index, row in station_info.iterrows():
    # Extract station name, latitude, and longitude from the current row
    station_name = row['station_name']
    latitude = row['latitude']
    longitude = row['longitude']
    
    # Print the name of the current station being processed
    print(f"Processing data for station: {station_name}")
    
    # Retrieve MODIS LAI data for the current station
    modis_lai = get_modisLAI_for_station(station_name=station_name)
    
    # Retrieve Copernicus LAI data for the current station using latitude and longitude
    copernicus_lai = get_copernicusLAI_for_station(longitude=longitude, latitude=latitude)
    
    # Save MODIS LAI data to a CSV file with a filename based on the station name
    modis_lai.to_csv(os.path.join(path_for_processed_modis_lai, f"{station_name}_modis.csv"))
    
    # Save Copernicus LAI data to a CSV file with a filename based on the station name
    copernicus_lai.to_csv(os.path.join(path_for_processed_copernicus_lai, f"{station_name}_copernicus.csv"))

# Print completion message
print("Processing complete.")

