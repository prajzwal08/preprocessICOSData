"""
Purpose:
This script processes meteorological data from two sources: ERA5-Land and in-situ measurements. 
It performs the following tasks:
1. Reads and preprocesses the data from CSV files.
2. Converts units and calculates additional variables such as vapor pressure deficit (VPD) and wind speed.
3. Renames columns and filters data to ensure consistency between the two datasets.
4. Stores unit information as metadata in the DataFrames and saves the cleaned data to CSV files.

Dependencies:
- pandas
- numpy
- matplotlib
- unit_conversion (custom module for unit conversions)

Usage:
Run this script as a standalone program. It will read input data from specified paths, process it, and save the cleaned data to new CSV files.

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional
from unit_conversion import kelvin_to_celsius, pascal_to_hectoPascal, meter_to_milimeter, joule_to_watt
from utils import calculate_vapor_pressure_deficit_from_temperatures,calculate_wind_speed_from_u_v



if __name__ == "__main__":
    era5_land_path = "~/data/fluxsites_NL/incoming/veenkampen/era5land_gee/era5land_veenkampen_final.csv"
    fluxnet_data_path = "~/data/fluxsites_NL/incoming/veenkampen/insitu_major_meteorological_variable.csv"
    intermediate_data_path = "~/data/fluxsites_NL/incoming/veenkampen/intermediate"

    # Read data
    df_era5_land = pd.read_csv(era5_land_path, index_col=0)
    df_insitu = pd.read_csv(fluxnet_data_path)
    
    # Ensure 'time' is in datetime format
    df_era5_land['time'] = pd.to_datetime(df_era5_land['time'])
    df_insitu['time'] = pd.to_datetime(df_insitu['time'])
    
    # Convert units for in-situ data
    df_insitu['air_temperature_degreeC'] = kelvin_to_celsius(np.array(df_insitu['air_temperature'].values))
    df_insitu['VPD_hPa'] = pascal_to_hectoPascal(np.array(df_insitu['VPD'].values))
    
    # Convert units for ERA5-Land data
    df_era5_land['total_precipitation_mm'] = meter_to_milimeter(df_era5_land['total_precipitation'].values)
    df_era5_land['air_temperature_degreeC'] = kelvin_to_celsius(np.array(df_era5_land['temperature_2m'].values))
    df_era5_land['dewpoint_temperature_degreeC'] = kelvin_to_celsius(np.array(df_era5_land['dewpoint_temperature_2m'].values))
    #Calculate vpd
    df_era5_land['VPD_hPa'] = calculate_vapor_pressure_deficit_from_temperatures(
        air_temperatures=df_era5_land['air_temperature_degreeC'].values,
        dew_point_temperatures=df_era5_land['dewpoint_temperature_degreeC'].values
    )
    df_era5_land['surface_solar_radiation_downwards_w_per_sqm'] = joule_to_watt(
        variable_joule=df_era5_land['surface_solar_radiation_downwards'].values,
        accumulation_period=1,
        unit="h"
    )
    df_era5_land['surface_thermal_radiation_downwards_w_per_sqm'] = joule_to_watt(
        variable_joule=df_era5_land['surface_thermal_radiation_downwards'].values,
        accumulation_period=1,
        unit="h"
    )
    df_era5_land['wind_speed'] = calculate_wind_speed_from_u_v(
        u_component_wind=df_era5_land['u_component_of_wind_10m'].values,
        v_component_wind=df_era5_land['v_component_of_wind_10m'].values
    )

    # Variables needed for in-situ data
    variables_needed_insitu = [
        'time',
        'P_1_1_7',
        'SW_IN_1_1_1',
        'LW_IN_1_1_1',
        'air_pressure',
        'air_temperature_degreeC',
        'VPD_hPa',
        'wind_speed',
    ]
    
    rename_mapping_insitu = {
        'P_1_1_7': 'precipitation',
        'SW_IN_1_1_1': 'swdown',
        'LW_IN_1_1_1': 'lwdown',
        'air_temperature_degreeC': 'air_temperature',
        'VPD_hPa': 'VPD'
    }
    
    df_insitu_selected = df_insitu[variables_needed_insitu].rename(columns=rename_mapping_insitu).iloc[1:,]
    
    units = {
        'precipitation': 'mm',
        'swdown': 'watt_per_sqm',
        'lwdown': 'watt_per_sqm',
        'air_pressure': 'pascal',
        'air_temperature': 'degreeC',
        'VPD': 'hPa',
        'wind_speed': 'm_per_s'
    }
    
    # Store unit information as metadata in the DataFrame
    df_insitu_selected.attrs['units'] = units
    df_insitu_selected.to_csv(os.path.join(intermediate_data_path, "cleaned_insitu_data_for_gap_filling.csv"))

    # Variables needed for ERA5-Land data
    variables_needed_era5_land = [
        'time',
        'total_precipitation_mm',
        'surface_solar_radiation_downwards',
        'surface_thermal_radiation_downwards',
        'surface_pressure',
        'air_temperature_degreeC',
        'VPD_hPa',
        'wind_speed',
    ]
    
    rename_mapping_era5_land = {
        'total_precipitation_mm': 'precipitation',
        'surface_solar_radiation_downwards': 'swdown',
        'surface_thermal_radiation_downwards': 'lwdown',
        'surface_pressure': 'air_pressure',
        'air_temperature_degreeC': 'air_temperature',
        'VPD_hPa': 'VPD'
    }

    # Convert the min and max time from df_insitu_selected to datetime64
    min_time = df_insitu_selected['time'].min()
    max_time = df_insitu_selected['time'].max()

    # Filter df_era5_land based on the time range from df_insitu_selected
    df_era5_land_selected = df_era5_land[
        (df_era5_land['time'] >= min_time) & 
        (df_era5_land['time'] <= max_time)
    ]

    # Select only the needed variables and rename the columns
    df_era5_land_selected = df_era5_land_selected[variables_needed_era5_land].rename(columns=rename_mapping_era5_land)
    
    df_era5_land_selected.attrs['units'] = units
    df_era5_land_selected.to_csv(os.path.join(intermediate_data_path, "cleaned_era5land_data_for_gap_filling.csv"))
