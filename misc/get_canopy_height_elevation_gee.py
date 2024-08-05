import ee
import pandas as pd
import os
import xarray as xr
import logging

os.chdir("/home/khanalp/code/PhD/preprocessICOSdata")

input_location = "csvs"
input_file = os.path.join(os.getcwd(),input_location,"00_stationinfo.csv")
output_file = os.path.join(os.getcwd(),input_location,"01_stationinfo_canopyht_elevation_from_gee.csv")
# Set up logging
log_file = os.path.join(os.getcwd(), 'logs','get_canopyheight_elevation.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ee.Authenticate() # This will give you the link, open it in the home browser and then copy the code to authenticate. 
# Some initial steps with google cloud CLI needs to be done if you are doing for the first time in remote machine. 
# Try searching how to use gee in the remote server. 

ee.Initialize()
logging.info("Earth Engine authenticated and initialized.")

stations = pd.read_csv(input_file)
logging.info(f"Loaded station data from {input_file}")

# Initialize an empty dictionary to store station_name, latitude, and longitude
station_coordinates = {}

# Iterate over each row in the DataFrame
for index, row in stations.iterrows():
    # Extract the station_name and position
    station_name = row['station_name']
    latitude = row['latitude']
    longitude = row['longitude']

    # Store the station_name, latitude, and longitude in the dictionary
    station_coordinates[station_name] = {'latitude': latitude, 'longitude': longitude}
    
logging.info("Station coordinates extracted.")


# Create an image collection for MODIS LAI data
dem_collection = ee.ImageCollection("COPERNICUS/DEM/GLO30") 
canopy_height = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1')
sd_canopy = ee.Image('users/nlang/ETH_GlobalCanopyHeightSD_2020_10m_v1')

# Initialize lists to store data
station_names = []
latitudes = []
longitudes = []
elevations = []
height_canopies = []
sd_height_canopies = []

# Iterate over each station in the dictionary
for station_name, coordinates in station_coordinates.items():
    
    # Extract latitude and longitude for the station
    latitude = coordinates['latitude']
    longitude = coordinates['longitude']
    
    # Define the location (longitude, latitude)
    point = ee.Geometry.Point(longitude, latitude)

    #Just because for some stations, elevations are not available. 
    try:
        # Get dem
        dem = dem_collection.filterBounds(point).first()
        # Get the elevation value at the specified point
        elevation = dem.sample(point, 30).first().get('DEM').getInfo()
        logging.info(f"Elevation for station '{station_name}': {elevation}")
    except Exception as e:
        logging.error(f"Error occurred for station '{station_name}' while getting elevation: {str(e)}")
        elevation = None
    
    try:
        # Sample the canopy height image at the target location
        height_canopy = canopy_height.sample(point, 10).first().get('b1').getInfo()
        logging.info(f"Canopy height for station '{station_name}': {height_canopy}")
    except Exception as e:
        logging.error(f"Error occurred for station '{station_name}' while sampling canopy height: {str(e)}")
        height_canopy = None
    
    try:
        # Sample the standard deviation of canopy height image at the target location
        sd_height_canopy = sd_canopy.sample(point, 10).first().get('b1').getInfo()
        logging.info(f"Standard deviation of canopy height for station '{station_name}': {sd_height_canopy}")
    except Exception as e:
        logging.error(f"Error occurred for station '{station_name}' while sampling standard deviation of canopy height: {str(e)}")
        sd_height_canopy = None
    
    # Append data to lists
    station_names.append(station_name)
    latitudes.append(latitude)
    longitudes.append(longitude)
    elevations.append(elevation)
    height_canopies.append(height_canopy)
    sd_height_canopies.append(sd_height_canopy)

logging.info("Data collection completed.")   

# Create DataFrame
df = pd.DataFrame({
    'station_name': station_names,
    'latitude': latitudes,
    'longitude': longitudes,
    'elevation': elevations,
    'height_canopy': height_canopies,
    'sd_height_canopy': sd_height_canopies
})

#Save this csv
df.to_csv(output_file, index=False)
logging.info(f"Data saved to {output_file}")



