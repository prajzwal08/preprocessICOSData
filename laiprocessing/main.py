# This script processes MODIS and Copernicus LAI data for different stations by retrieving and saving data 
# to CSV files only if the data for a station does not already exist in the specified directories.

# Import necessary libraries
import os
import pandas as pd

# Importing functions to retrieve MODIS and Copernicus LAI data for a given station
from preprocessmodisLAI import get_modisLAI_for_station
from preprocesscopernicuslai import get_copernicusLAI_for_station

# Define file paths for station information and processed LAI data
file_path_station_info = "/home/khanalp/code/PhD/preprocessICOSdata/csvs/02_station_with_elevation_heightcanopy.csv"
path_for_processed_modis_lai = "/home/khanalp/data/processed/lai/modis"
path_for_processed_copernicus_lai = "/home/khanalp/data/processed/lai/copernicus/v0/processed"  # Change to v0 for v0

# Read station information from a CSV file into a pandas DataFrame
df_station_details = pd.read_csv(file_path_station_info)

# Print header/title
print("Processing MODIS and Copernicus LAI Data")

# Iterate over each row in the station information DataFrame
for index, row in df_station_details.iterrows():
    # Extract station name, latitude, and longitude from the current row
    station_name = row['station_name']
    latitude = row['latitude']
    longitude = row['longitude']
    
    # Print the name of the current station being processed
    print(f"Processing data for station: {station_name}")
    
    # Construct file paths for MODIS and Copernicus data
    modis_file_path = os.path.join(path_for_processed_modis_lai, f"{station_name}_modis.csv")
    copernicus_file_path = os.path.join(path_for_processed_copernicus_lai, f"{station_name}_copernicus.csv")

    # Check if the MODIS LAI file already exists
    if not os.path.exists(modis_file_path):
        # Retrieve MODIS LAI data for the current station
        print(f"Retrieving MODIS LAI data for {station_name}...")
        modis_lai = get_modisLAI_for_station(station_name=station_name)
        # Save MODIS LAI data to a CSV file
        modis_lai.to_csv(modis_file_path)
    else:
        print(f"MODIS LAI data for {station_name} already exists. Skipping...")

    # Check if the Copernicus LAI file already exists
    if not os.path.exists(copernicus_file_path):
        # Retrieve Copernicus LAI data for the current station using latitude and longitude
        print(f"Retrieving Copernicus LAI data for {station_name}...")
        copernicus_lai = get_copernicusLAI_for_station(longitude=longitude, latitude=latitude)
        # Save Copernicus LAI data to a CSV file
        copernicus_lai.to_csv(copernicus_file_path)
    else:
        print(f"Copernicus LAI data for {station_name} already exists. Skipping...")

# Print completion message
print("Processing complete.")
