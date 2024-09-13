"""
This script processes meteorological data from two sources: ERA5-Land and in-situ measurements.
It performs the following tasks:

1. Reads and preprocesses the data from CSV files.
2. Converts units and calculates additional variables such as vapor pressure deficit (VPD) and wind speed.
3. Resamples in-situ data to match ERA5-Land hourly intervals.
4. Filters and selects relevant variables, aligning time ranges between the datasets.
5. Merges ERA5-Land and in-situ data for both instantaneous and accumulative variables. Accumulative variables are precipitation,ssrd and strd in the ERA5land. 
6. Adds unit information as metadata to the DataFrames.
7. Reorders columns and saves the cleaned, combined data to new CSV files.
"""

import os
import pandas as pd
from unit_conversion import (
    kelvin_to_celsius, pascal_to_hectoPascal, meter_to_millimeter,
    joule_to_watt, kilopascal_to_hectoPascal, convert_local_to_utc
)
from utils import (
    calculate_vapor_pressure_deficit_from_temperatures, 
    calculate_wind_speed_from_u_v, 
    convert_accumulated_values_to_hourly_values
)

def resample_variables(df_insitu_accumulative: pd.DataFrame, accumulative_variables: list) -> pd.DataFrame:
    """
    Resample in-situ accumulative variables to match ERA5-Land hourly intervals.

    Args:
        df_insitu_accumulative (pd.DataFrame): In-situ data with accumulative variables.
        accumulative_variables (list): List of accumulative variables to resample.

    Returns:
        pd.DataFrame: Resampled DataFrame with hourly data.
    """
    resampled_dict = {}
    
    for variable in accumulative_variables:
        if variable == "precipitation":
            resampled_df = df_insitu_accumulative.resample('1H', closed='right', label='right').sum()
        else:
            resampled_df = df_insitu_accumulative.resample('1H', closed='right', label='right').mean()
        resampled_dict[variable] = resampled_df[variable]
    
    # Combine all resampled DataFrames into one
    combined_resampled_df = pd.concat(resampled_dict.values(), axis=1)
    combined_resampled_df.columns = accumulative_variables
    
    return combined_resampled_df

def process_era5_land_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process ERA5-Land data including unit conversions and calculations.

    Args:
        df (pd.DataFrame): ERA5-Land data.

    Returns:
        pd.DataFrame: Processed ERA5-Land DataFrame.
    """
    df['air_temperature_degreeC'] = kelvin_to_celsius(df['temperature_2m'].values)
    df['dewpoint_temperature_degreeC'] = kelvin_to_celsius(df['dewpoint_temperature_2m'].values)
    
    df['VPD_hPa'] = kilopascal_to_hectoPascal(calculate_vapor_pressure_deficit_from_temperatures(
        air_temperatures=df['air_temperature_degreeC'].values,
        dew_point_temperatures=df['dewpoint_temperature_degreeC'].values
    ))
    df['surface_solar_radiation_downwards_hourly'] = convert_accumulated_values_to_hourly_values(df['surface_solar_radiation_downwards'].values)
    df['surface_thermal_radiation_downwards_hourly'] = convert_accumulated_values_to_hourly_values(df['surface_thermal_radiation_downwards'].values)
    df['precipitation_hourly'] = convert_accumulated_values_to_hourly_values(df['total_precipitation'].values)
    df['hourly_surface_solar_radiation_downwards_w_per_sqm'] = joule_to_watt(
        df['surface_solar_radiation_downwards_hourly'].values, 
        accumulation_period=1, 
        unit="h"
    )
    df['hourly_surface_thermal_radiation_downwards_w_per_sqm'] = joule_to_watt(
        df['surface_thermal_radiation_downwards_hourly'].values, 
        accumulation_period=1, 
        unit="h"
    )
    df['hourly_precipitation_mm'] = meter_to_millimeter(df['precipitation_hourly'].values)
    df['wind_speed'] = calculate_wind_speed_from_u_v(
        u_component_wind=df['u_component_of_wind_10m'].values,
        v_component_wind=df['v_component_of_wind_10m'].values
    )
    return df

def process_insitu_data(df: pd.DataFrame, time_difference: int) -> pd.DataFrame:
    """
    Process in-situ data including unit conversions and time adjustments.

    Args:
        df (pd.DataFrame): In-situ data.
        time_difference (int): Time difference between local and UTC.

    Returns:
        pd.DataFrame: Processed in-situ DataFrame.
    """
    df['air_temperature_degreeC'] = kelvin_to_celsius(df['air_temperature'].values)
    df['VPD_hPa'] = pascal_to_hectoPascal(df['VPD'].values)
    df['utc_time'] = convert_local_to_utc(df['time'].values, time_difference=time_difference)
    
    return df

def main():
    # File paths
    era5_land_path = "~/data/fluxsites_NL/incoming/veenkampen/era5land_gee/era5land_veenkampen_final.csv"
    fluxnet_data_path = "~/data/fluxsites_NL/incoming/veenkampen/insitu_major_meteorological_variable.csv"
    intermediate_data_path = "~/data/fluxsites_NL/incoming/veenkampen/intermediate"
    
    time_difference = +1
    try:
        df_era5_land = pd.read_csv(era5_land_path, index_col=0)
        df_insitu = pd.read_csv(fluxnet_data_path)
        
        # Process ERA5-Land and in-situ data
        df_era5_land = process_era5_land_data(df_era5_land)
        df_insitu = process_insitu_data(df_insitu, time_difference)
        
        # Define start_time_insitu 
        start_time = pd.to_datetime("2015-01-01 00:30:00")            
        # Define variables
        instantaneous_variable = ['air_pressure', 'air_temperature', 'VPD', 'wind_speed']
        accumulative_variable = ['precipitation', 'swdown', 'lwdown']
        
        # Variables needed for in-situ data
        variables_needed_insitu = [
            'utc_time',
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
        
        # Variables needed for ERA5-Land data
        variables_needed_era5_land = [
            'time',
            'hourly_precipitation_mm',
            'hourly_surface_solar_radiation_downwards_w_per_sqm',
            'hourly_surface_thermal_radiation_downwards_w_per_sqm',
            'surface_pressure',
            'air_temperature_degreeC',
            'VPD_hPa',
            'wind_speed',
        ]
        
        rename_mapping_era5_land = {
            'hourly_precipitation_mm': 'precipitation',
            'hourly_surface_solar_radiation_downwards_w_per_sqm': 'swdown',
            'hourly_surface_thermal_radiation_downwards_w_per_sqm': 'lwdown',
            'surface_pressure': 'air_pressure',
            'air_temperature_degreeC': 'air_temperature',
            'VPD_hPa': 'VPD'
        }
        
        units = {
        'precipitation': 'mm',
        'swdown': 'watt_per_sqm',
        'lwdown': 'watt_per_sqm',
        'air_pressure': 'pascal',
        'air_temperature': 'degreeC',
        'VPD': 'hPa',
        'wind_speed': 'm_per_s'
        }
    
        
        # Select variables needed and rename them. 
        df_insitu_selected = df_insitu[variables_needed_insitu].rename(columns=rename_mapping_insitu)
        # Adjust negative values for specific variables
        df_insitu_selected.loc[df_insitu_selected['swdown'] < 0, 'swdown'] = 0
        # Make sure they start from 00:30
        df_insitu_selected = df_insitu_selected[ (df_insitu_selected['utc_time'] >= start_time)]
        df_insitu_selected['utc_time'] = pd.to_datetime(df_insitu_selected['utc_time'] )
        # Store unit information as metadata in the DataFrame
        df_insitu_selected.attrs['units'] = units
    
        
        # Select variables needed and rename them 
        df_era5_land_selected = df_era5_land[variables_needed_era5_land].rename(columns=rename_mapping_era5_land)
        df_era5_land_selected['time'] = pd.to_datetime(df_era5_land_selected['time'] )
        # Filter df_era5_land based on the time range from df_insitu_selected
        df_era5_land_selected = df_era5_land_selected[(df_era5_land_selected['time'] > start_time) & (df_era5_land_selected['time'] <= df_insitu_selected['utc_time'].max())]
        df_era5_land_selected.attrs['units'] = units
        
        
        # Merge data for instantaneous variables
        merged_df_instantaneous = pd.merge(
            df_era5_land_selected[instantaneous_variable + ["time"]],
            df_insitu_selected[instantaneous_variable + ["utc_time"]],
            left_on='time',
            right_on='utc_time',
            suffixes=('_era5land', '_insitu')
        ).drop('time', axis=1)
        
        # Process and resample accumulative variables for insitu to change from 30 mins to hourly 
        df_insitu_accumulative = df_insitu_selected[accumulative_variable + ["utc_time"]]
        df_insitu_accumulative.set_index('utc_time', inplace=True)
        
        df_resampled = resample_variables(df_insitu_accumulative, accumulative_variable)
        df_resampled = df_resampled.reset_index()
        
        merged_df_accumulative = pd.merge(
            df_era5_land_selected[accumulative_variable+ ["time"]],
            df_insitu_selected[accumulative_variable + ["utc_time"]],
            left_on='time',
            right_on='utc_time',
            suffixes=('_era5land', '_insitu')
        ).drop('time', axis=1)
        
        combined_df = pd.merge(merged_df_instantaneous, merged_df_accumulative, on='utc_time')
        
        # Reorder columns
        utc_time_col = 'utc_time'
        era5land_cols = [col for col in combined_df.columns if col.endswith('_era5land')]
        insitu_cols = [col for col in combined_df.columns if col.endswith('_insitu')]
        ordered_cols = [utc_time_col] + era5land_cols + insitu_cols
        combined_df = combined_df[ordered_cols]
        combined_df.attrs['units'] = units
        
        combined_df.to_csv(os.path.join(intermediate_data_path,"cleaned_data_insitu_era5land.csv"))
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
