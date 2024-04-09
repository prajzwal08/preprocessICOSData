"""
Author: [Prajwal Khanal]
Date: [Thursday, 21 March 2024 @ Boston Public Library @ 5:34 PM]
Purpose: This script preprocesses ICOS data, including selecting, renaming, and converting variables,
calculating additional variables like LAI, CO2, relative humidity, and specific humidity, 
and saving the processed data to NetCDF files.
"""

# importing libraries
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

#importing modules
from utils import list_folders_with_prefix, list_csv_files_in_folder,read_csv_file_with_station_name
from utils import select_rename_convert_to_xarray
from utils import calculate_relative_and_specific_humidity
from laiprocessing.preprocessLAI import get_LAI_for_station
from getco2 import get_co2_data

# Set the working directory
os.chdir("/home/khanalp/code/PhD/preprocessICOSdata")

if __name__ == "__main__":   
    # define  file paths
    modis_path = "/home/khanalp/data/MODIS_Raw/"
    cams_path = "/home/khanalp/data/cams/cams_europe_2003_2020.nc"
    ICOS_location = "/home/khanalp/data/ICOS2020"
    output_directory = '/home/khanalp/data/processed/input_pystemmus'
    prefix = "FLX"
    # For these stations, I want to process input data only until 2019.
    station_with_missing_2020data_path = "/home/khanalp/code/PhD/preprocessICOSdata/output/csvs/overview_missingdata_for_station.csv"
    
    # Define alternative start date, the start_date cannot be before 2003 because of non-availability of cams and MODIS LAI.
    alternative_start_date = np.datetime64('2003-01-01 00:00:00')
    
    # Read station information
    station_all = pd.read_csv("/home/khanalp/output/csvs/stationdetails.csv") #This contains all the station details. 
    station_all = station_all.set_index('station_name')
    station_all.columns = [col.replace(' ', '_') if ' ' in col else col for col in station_all.columns] # Replace spaces with underscores in column names where spaces are not present
    
    # Create a mask for rows where 'height_canopy_field_information' is not NaN
    mask = station_all['height_canopy_field_information'].notna()
    station_all.loc[mask, 'height_canopy'] = station_all.loc[mask, 'height_canopy_field_information'] # Assign values using .loc[]
    station_all.loc[~mask, 'height_canopy'] = station_all.loc[~mask, 'height_canopy_ETH']

    # Read the list of stations with missing 2020 data
    df_station_with_missing_2020 = pd.read_csv(station_with_missing_2020data_path)
    station_with_missing_2020_list =df_station_with_missing_2020['station_name'].unique().tolist()
    
    
    # List folders and CSV files
    folders = list_folders_with_prefix(ICOS_location, prefix)
    csv_files = []
    for folder in folders:
        folder_path = os.path.join(ICOS_location, folder)
        csv_files.extend(list_csv_files_in_folder(folder_path, "FULLSET_HH"))


    # Process data for each station
    for index, station_info in station_all.iterrows():
        print(index)
        # Read corresponding csv file for ICOS station. 
        data_frame = read_csv_file_with_station_name(index,csv_files)
        
        if data_frame is None:
            print(f"CSV file not found for station {index}. Skipping...")
            continue
    
        #selecting required variables from ICOS data for input.
        selected_variables = [
            'TIMESTAMP_START',
            'TA_F',
            'TA_F_QC',
            'SW_IN_F',
            'SW_IN_F_QC',
            'LW_IN_F',
            'LW_IN_F_QC',
            'VPD_F',
            'VPD_F_QC',
            'PA_F',
            'PA_F_QC',
            'P_F',
            'P_F_QC',
            'WS_F',
            'WS_F_QC',
            'RH',
            'CO2_F_MDS',
            'CO2_F_MDS_QC' 
        ]

        #Renaming them 
        rename_mapping = {
            'TA_F':'Tair',
            'TA_F_QC':'Tair_qc',
            'SW_IN_F':'SWdown',
            'SW_IN_F_QC':'SWdown_qc',
            'LW_IN_F':'LWdown',
            'LW_IN_F_QC':'LWdown_qc',
            'VPD_F':'VPD',
            'VPD_F_QC':'VPD_qc',
            'PA_F':'Psurf',
            'PA_F_QC':'Psurf_qc',
            'P_F' : 'Precip',
            'P_F_QC':'Precip_qc',
            'WS_F':'Wind',
            'WS_F_QC':'Wind_qc',
            'CO2_F_MDS':'CO2air',
            'CO2_F_MDS_QC':'CO2air_qc'
            }

        # Get xarray with selected variables and after renaming .
        data_xarray = select_rename_convert_to_xarray(data_frame,selected_variables,rename_mapping)
        
        #getting other  variables in xarray.
        data_xarray['latitude'] = xr.DataArray(np.array(station_info['latitude']).reshape(1,-1), dims=['x','y'])
        data_xarray['longitude'] = xr.DataArray(np.array(station_info['longitude']).reshape(1,-1), dims=['x','y'])
        data_xarray['reference_height'] = xr.DataArray(np.array(station_info['Measurement_height']).reshape(1,-1), dims=['x','y'])
        data_xarray['canopy_height'] = xr.DataArray(np.array(station_info['height_canopy']).reshape(1,-1), dims=['x','y'])
        data_xarray['elevation'] = xr.DataArray(np.array(station_info['elevation']).reshape(1,-1), dims=['x','y'])
        data_xarray['IGBP_veg_short'] = xr.DataArray(np.array(station_info['IGBP_short_name'], dtype = 'S200'))
        data_xarray['IGBP_veg_long'] = xr.DataArray(np.array(station_info['IGBP_long_name'], dtype = 'S200'))


        #Count no of nas, which are -9999 in flux data.
        counts = {var: np.count_nonzero(data_xarray[var].values == -9999) for var in data_xarray.data_vars}
        
        # If start date is before 2003, make 2003 as start. 
        if data_xarray.time.values.min() < alternative_start_date:
            data_xarray = data_xarray.sel(time=slice(alternative_start_date, None))
            
        # for station with missing variables in 2020, i want to truncate it until 2019.
        if str(index) in str(station_with_missing_2020_list):
            data_xarray = data_xarray.sel(time=slice(None, '2019-12-31 23:30:00'))
        
        ## Processing for getting LAI and CAMS co2
        #-----------------------------------------
        # Get the time values
        time_values = data_xarray.time.values
        # Calculate the difference between the first and second time values
        time_difference_seconds = (time_values[1] - time_values[0]).astype('timedelta64[s]')
        # Convert the time difference to minutes
        time_difference_minutes = time_difference_seconds / np.timedelta64(1, 'm')
        # Format the time difference as "30min" string
        time_difference_string = f"{int(time_difference_minutes)}min"
        # ----------------------
        
        # # getting LAI    
        lai = get_LAI_for_station(modis_path = modis_path,
                                station_name= index,
                                start_date=data_xarray.time.values.min(),
                                end_date=data_xarray.time.values.max(),
                                time_interval=time_difference_string
                                )
        data_xarray['LAI'] = xr.DataArray(lai.reshape(1,1,-1), dims=['x','y','time']) # Add LAI variable.
        data_xarray['LAI_alternative'] = xr.DataArray(lai.reshape(1,1,-1), dims=['x','y','time'])
        
        # If CO2air is missing, get from cams data.
        if counts['CO2air'] > 0:
            co2_array = get_co2_data(latitude= data_xarray['latitude'].values.flatten(),
                                                longitude= data_xarray['longitude'].values.flatten(),
                                                start_time = data_xarray.time.values.min(),
                                                end_time = data_xarray.time.values.max(),
                                                resampling_interval=time_difference_string,
                                                file_path=cams_path
                                                )
            data_xarray['CO2air'] = xr.DataArray(co2_array.reshape(1,1,-1), dims=['x','y','time'])
            attributes_CO2air = {'method':'from cams, due to insufficient field record'} # Write attrs to know where it comes from.
            data_xarray['CO2air'].attrs.update(attributes_CO2air)
        else: 
            attributes_CO2air = {'method':'from field '}
            data_xarray['CO2air'].attrs.update(attributes_CO2air)
        
        
        # Calculate relative and specific humidity 
        relative_humidity, specific_humidity = calculate_relative_and_specific_humidity(data_xarray.VPD.values.flatten(), 
                                                                                        data_xarray.Tair.values.flatten(), 
                                                                                        data_xarray .Psurf.values.flatten())
        
        if counts['RH'] > 0: #IF missing, replace with calculated values, if not use filed value.
            data_xarray['Qair'] = xr.DataArray(specific_humidity.reshape(1,1,-1), dims=['x','y','time'])
            data_xarray['RH'] = xr.DataArray(relative_humidity.reshape(1,1,-1), dims=['x','y','time']) 
            #Updating attibutes
            attributes_RH_Qair = {'method':'calculated from VPD,Tair,Psurf, ignoring field data.'}
            data_xarray['Qair'].attrs.update(attributes_RH_Qair)
            data_xarray['RH'].attrs.update(attributes_RH_Qair)
        else: 
            #Updating attibutes
            data_xarray['Qair'] = xr.DataArray(specific_humidity.reshape(1,1,-1), dims=['x','y','time'])
            attributes_RH_Qair = {'method':'calculated from VPD,Tair,Psurf, ignoring field data.'}
            data_xarray['Qair'].attrs.update(attributes_RH_Qair)
            
        # Changing the units (for other, they are in the unit same as Plumber2)
        data_xarray['Precip'] = data_xarray['Precip'] / (30*60) # changing from mm of 30 mins to mm/s.  (kg/sq.m/s = mm/s)
        data_xarray['Tair'] = data_xarray['Tair'] + 273.15 # degree C to kelvin
        data_xarray['Psurf'] = data_xarray['Psurf']*1000 #kPa to Pa
        
        
        # Ensuring all variables in float32 format. 
        for var_name in data_xarray.data_vars:
            if var_name not in ['IGBP_veg_short', 'IGBP_veg_long']:
                data_xarray[var_name] = data_xarray[var_name].astype('float32')
            
        #Checking for missing values and printing. 
        # Filter variables without '_qc' in their names
        data_vars_without_qc = [var_name for var_name in data_xarray.data_vars if '_qc' not in var_name]
        
        # Initialize a flag to track if there are any missing values
        no_missing_values = True
        
        # Check for missing values in each variable
        for var_name in data_vars_without_qc:
            has_missing_values = data_xarray[var_name].isnull().any()
            if has_missing_values:
                print(f"Variable '{var_name}' has missing values.")
                no_missing_values = False
                break

        # If there are no missing values, save the Dataset
        if no_missing_values:
            # Extract the start and end years from the time coordinates
            # Convert time coordinates to pandas datetime format
            time_pd = pd.to_datetime(data_xarray.time.values)

            # Extract the start and end years from the time coordinates
            start_year = time_pd.min().year
            end_year = time_pd.max().year
            # Construct the filename
            filename = f"FLX_{index}_FLUXNET2015_FULLSET_{start_year}-{end_year}.nc"
            output_path = os.path.join(output_directory,filename)
            data_xarray.to_netcdf(output_path)
            print("Dataset saved successfully.")
        else:
            print("There are missing values in one or more variables. Dataset not saved.")
                
        

#Just to check the data.
# trial = xr.open_dataset("/home/khanalp/data/processed/input_pystemmus/FLX_BE-Maa_FLUXNET2015_FULLSET_2016-2019.nc")
# # Iterate through each data variable
# for var_name in trial.data_vars:
#     # Print variable name
#     print(f"Variable: {var_name}")
    
#     # Print variable attributes
#     print(f"Attributes: {trial[var_name].attrs}")
    
#     # Add more attributes as needed
    
#     # Print a separator for clarity
#     print("-" * 20)
