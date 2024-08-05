import ee
import pandas as pd
import os
import logging

"""
Purpose:
This script processes a CSV file containing station information to extract canopy height and elevation data using Google Earth Engine (GEE).
It performs the following tasks:
1. Authenticates and initializes the Google Earth Engine.
2. Loads station data from a CSV file.
3. Retrieves elevation data and canopy height information from GEE for each station.
4. Logs the data collection process, including any errors encountered.
5. Saves the collected data into a new CSV file for further analysis.
"""

# Set working directory
os.chdir("/home/khanalp/code/PhD/preprocessICOSdata")

# Define file paths
input_location = "csvs"
input_file = os.path.join(os.getcwd(), input_location, "00_stationinfo.csv")
output_file = os.path.join(os.getcwd(), input_location, "01_stationinfo_canopyht_elevation_from_gee.csv")

# Set up logging
log_file = os.path.join(os.getcwd(), 'logs', 'get_canopyheight_elevation.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Authenticate and initialize Google Earth Engine (GEE)
ee.Authenticate()  # Authenticate using Google Earth Engine credentials
ee.Initialize()    # Initialize the Earth Engine library

logging.info("Earth Engine authenticated and initialized.")

# Load station data from CSV
stations = pd.read_csv(input_file)
logging.info(f"Loaded station data from {input_file}")

# Initialize a dictionary to store station coordinates
station_coordinates = {}

# Populate the dictionary with station names and their coordinates
for index, row in stations.iterrows():
    station_name = row['station_name']
    latitude = row['latitude']
    longitude = row['longitude']
    station_coordinates[station_name] = {'latitude': latitude, 'longitude': longitude}

logging.info("Station coordinates extracted.")

# Define Earth Engine image collections and images
dem_collection = ee.ImageCollection("COPERNICUS/DEM/GLO30")  # Digital Elevation Model (DEM)
canopy_height = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1')  # Canopy height data
sd_canopy = ee.Image('users/nlang/ETH_GlobalCanopyHeightSD_2020_10m_v1')  # Standard deviation of canopy height

# Initialize lists to store data for each station
station_names = []
latitudes = []
longitudes = []
elevations = []
height_canopies = []
sd_height_canopies = []

# Process each station to extract elevation, canopy height, and standard deviation
for station_name, coordinates in station_coordinates.items():
    latitude = coordinates['latitude']
    longitude = coordinates['longitude']
    
    # Define the geographic point for the station
    point = ee.Geometry.Point(longitude, latitude)

    try:
        # Get the DEM image for the location and extract elevation value
        dem = dem_collection.filterBounds(point).first()
        elevation = dem.sample(point, 30).first().get('DEM').getInfo()
        logging.info(f"Elevation for station '{station_name}': {elevation}")
    except Exception as e:
        logging.error(f"Error occurred for station '{station_name}' while getting elevation: {str(e)}")
        elevation = None
    
    try:
        # Sample canopy height image at the specified point
        height_canopy = canopy_height.sample(point, 10).first().get('b1').getInfo()
        logging.info(f"Canopy height for station '{station_name}': {height_canopy}")
    except Exception as e:
        logging.error(f"Error occurred for station '{station_name}' while sampling canopy height: {str(e)}")
        height_canopy = None
    
    try:
        # Sample standard deviation of canopy height image at the specified point
        sd_height_canopy = sd_canopy.sample(point, 10).first().get('b1').getInfo()
        logging.info(f"Standard deviation of canopy height for station '{station_name}': {sd_height_canopy}")
    except Exception as e:
        logging.error(f"Error occurred for station '{station_name}' while sampling standard deviation of canopy height: {str(e)}")
        sd_height_canopy = None
    
    # Append the gathered data to the lists
    station_names.append(station_name)
    latitudes.append(latitude)
    longitudes.append(longitude)
    elevations.append(elevation)
    height_canopies.append(height_canopy)
    sd_height_canopies.append(sd_height_canopy)

logging.info("Data collection completed.")

# Create a DataFrame from the collected data
df = pd.DataFrame({
    'station_name': station_names,
    'latitude': latitudes,
    'longitude': longitudes,
    'elevation': elevations,
    'height_canopy': height_canopies,
    'sd_height_canopy': sd_height_canopies
})

# Save the DataFrame to a CSV file
df.to_csv(output_file, index=False)
logging.info(f"Data saved to {output_file}")
