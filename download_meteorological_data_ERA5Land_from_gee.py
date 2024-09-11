"""
Purpose:
This script retrieves meteorological data from the ECMWF ERA5-Land dataset for a specified FLUXNET site 
using Google Earth Engine (GEE). It processes the data in 6-month intervals from a start date to an end date,
for a list of required bands. The script performs the following tasks:

1. Authenticates and initializes Google Earth Engine.
2. Reads station details (latitude and longitude) from a CSV file.
3. Generates 6-month intervals between the specified start and end dates.
4. Retrieves and reduces data for each band within each interval.
5. Merges the data for each band based on the time stamp.
6. Saves the combined data to a CSV file.

Dependencies:
- `earthengine-api`: Google Earth Engine API for data access.
- `pandas`: For data manipulation and saving results.
- `os`: For file path operations.
"""

import ee
import pandas as pd
import os
from typing import List, Tuple

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize()

# Path variables
station_details = '/home/khanalp/code/PhD/preprocessICOSdata/csvs/02_station_with_elevation_heightcanopy.csv'
output_path = '/home/khanalp/data/fluxsites_NL/incoming/veenkampen/era5land_gee'

# Read station details from CSV
df_station_details = pd.read_csv(station_details)
station_name = "NL-Vee"
latitude = df_station_details.loc[df_station_details['station_name'] == station_name, 'latitude'].values[0]
longitude = df_station_details.loc[df_station_details['station_name'] == station_name, 'longitude'].values[0]

# Define start and end dates
start_date = "2015-01-01"
end_date = "2023-12-31"

def generate_six_month_intervals(start_date: str, end_date: str) -> List[Tuple[str, str]]:
    #Generate a list of 6-month intervals between the start and end dates. This is because of the 
    # request limit of 5000 in GEE.
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    intervals = []

    while start < end:
        period_end = (start + pd.DateOffset(months=6)) - pd.DateOffset(days=1)
        if period_end > end:
            period_end = end

        intervals.append((start.strftime('%Y-%m-%d'), period_end.strftime('%Y-%m-%d')))
        start = period_end + pd.DateOffset(days=1)

    return intervals

intervals = generate_six_month_intervals(start_date, end_date)

# List of bands to retrieve
bands_required = [
    'v_component_of_wind_10m', 
    'u_component_of_wind_10m',
    'dewpoint_temperature_2m',
    'temperature_2m',
    'surface_pressure',
    'surface_solar_radiation_downwards', 
    'surface_thermal_radiation_downwards',
    'total_precipitation'
]

point = ee.Geometry.Point(longitude, latitude)

def get_band_data_as_dataframe(band_name, point, start_date, end_date):
    # Retrieve data for a specific band and date range, and convert it to a pandas DataFrame.
    
    collection = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
                    .filterDate(start_date, ee.Date(end_date).advance(1, 'day')) \
                    .select(band_name)

    def reduce_image(image):
        reduced_value = image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=9000  # Scale matches the native resolution of the dataset
        ).get(band_name)
        timestamp = ee.Date(image.get('system:time_start'))
        return ee.Feature(None, {'time': timestamp, band_name: reduced_value})

    reduced_collection = collection.map(reduce_image)
    data_list = reduced_collection.getInfo()['features']
    data = [{'time': pd.to_datetime(f['properties']['time']['value'], unit='ms'),
             band_name: f['properties'][band_name]} for f in data_list]

    df = pd.DataFrame(data)
    return df

# Initialize dictionary to store DataFrames for each band
df_dict = {band: [] for band in bands_required}

# Retrieve data for each band
for band in bands_required:
    print(f'Processing band: {band}')
    for interval in intervals:
        print(f'Processing interval: {interval[0]} to {interval[1]}')
        df = get_band_data_as_dataframe(band, point, interval[0], interval[1])
        df_dict[band].append(df)

# Merge DataFrames for each band on the 'time' column
merged_df = pd.DataFrame()
for band, dfs in df_dict.items():
    band_df = pd.concat(dfs, ignore_index=True)
    if merged_df.empty:
        merged_df = band_df
    else:
        merged_df = pd.merge(merged_df, band_df, on='time', how='outer')

# Save the merged DataFrame to CSV
output_file_path = os.path.join(output_path, f'ERA5land_{station_name}_{start_date}_{end_date}.csv')
merged_df.to_csv(output_file_path, index=False)

print(f'Data saved to {output_file_path}')
