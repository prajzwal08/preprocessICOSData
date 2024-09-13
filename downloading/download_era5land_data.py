#Python client for the Copernicus Climate Data Store (CDS) API.
import cdsapi
import os
import pandas as pd


def calculate_extent(center_lat: float, center_lon: float, resolution: float = 0.1) -> dict:
    """
    Calculate the extent around a center point to cover a 3x3 grid of pixels.

    Parameters:
    center_lat (float): Latitude of the center point.
    center_lon (float): Longitude of the center point.
    resolution (float): Resolution of the data in degrees. Default is 0.1 degrees.

    Returns:
    dict: A dictionary with 'lat_min', 'lat_max', 'lon_min', 'lon_max' keys.
    """
    # Calculate the extent to cover a 3x3 grid around the center point
    grid_size = 1  # 1 degree to cover 3x3 grid (0.5 degree on each side)
    
    lat_min = center_lat - grid_size / 2
    lat_max = center_lat + grid_size / 2
    lon_min = center_lon - grid_size / 2
    lon_max = center_lon + grid_size / 2
    
    return {
        'lat_min': lat_min,
        'lat_max': lat_max,
        'lon_min': lon_min,
        'lon_max': lon_max
    }

if __name__ == "__main__":        
    #File path
    station_details = '/home/khanalp/code/PhD/preprocessICOSdata/csvs/02_station_with_elevation_heightcanopy.csv'  
    # Load station details from CSV
    df_station_details = pd.read_csv(station_details) 
    station_name = "NL-Vee"
    station_info = df_station_details[df_station_details['station_name'] == station_name]

    #Create an instance of CDS API client and assign it to c. 
    c = cdsapi.Client()

    # Define the data request parameters, these are parameters needed for STEMMUS-SCOPE run
    variables = ['10m_u_component_of_wind', '10m_v_component_of_wind','2m_dewpoint_temperature','2m_temperature',
                'surface_pressure', 'surface_solar_radiation_downwards', 'surface_thermal_radiation_downwards',
                'total_precipitation']

    # Define location and date
    # Example usage
    center_lat = station_info['latitude'].values
    center_lon = station_info['longitude'].values
    extent = calculate_extent(center_lat, center_lon)
    location = [float(extent['lat_min']),float(extent['lon_min']),float(extent['lat_max']),float(extent['lon_max'])] 

    start_year = 2015
    end_year = 2022


    #output location 
    output_location = "/home/khanalp/data/fluxsites_NL/incoming/veenkampen/era5land"

    #Looping through each variables
    for variable in variables:
    # Define the folder name to store downloaded data 
        folder_name = f'{output_location}/{variable}'
        

        # Create the full path to the folder in the working directory
        folder_path = os.path.join(os.getcwd(), folder_name)
        
        # Check if the folder exists, and if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        for year in range(start_year, end_year+1):
            request = {
                'product_type': 'reanalysis',
                'data_format': 'netcdf',
                'download_format': 'unarchived',
                'variable': variable,
                'year': str(year),
                'month': [f"{month:02d}" for month in range(1, 13)],
                'day': [f"{day:02d}" for day in range(1, 32)],
                'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00',
                        '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00',
                        '20:00', '21:00', '22:00', '23:00'],
                'area': location,  # Latitude and longitude for your location
            }

        # Request the data for the current variable
        output_filename = f'era5_land_{variable}_{year}.nc'
        c.retrieve('reanalysis-era5-land', request, os.path.join(folder_path, output_filename))

    
