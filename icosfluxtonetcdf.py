"""
Author: [Prajwal Khanal]
Date: [2024 May 3, Friday]

Description:
This script processes ICOS flux data stored in CSV files, converts it to xarray dataset,
assigns attributes to variables, and saves the dataset as NetCDF files.

Dependencies:
- xarray
- pandas
- numpy
- os
- [Other dependencies used in utils module]

"""

import os
import xarray as xr
import pandas as pd
from utils import list_folders_with_prefix, list_csv_files_in_folder, select_rename_convert_to_xarray

# Define paths
file_path_station_info = "/home/khanalp/code/PhD/preprocessICOSdata/output/csvs/stations_readyformodelrun.csv"
input_data_path_flux = "/home/khanalp/data/ICOS2020"
output_path = "/home/khanalp/data/processed/insituflux"

os.chdir("/home/khanalp/code/PhD/preprocessICOSdata")
# List folders and CSV files
prefix = "FLX"
folders = list_folders_with_prefix(input_data_path_flux, prefix)
csv_files = []
for folder in folders:
    folder_path = os.path.join(input_data_path_flux, folder)
    csv_files.extend(list_csv_files_in_folder(folder_path, "FULLSET_HH"))

variable_info = {
    'Rnet': {'rename_from': 'NETRAD', 'unit': 'W/m²', 'description': 'Net radiation', 'missing_value': -9999},
    'SWup': {'rename_from': 'SW_OUT', 'unit': 'W/m²', 'description': 'Upward shortwave radiation', 'missing_value': -9999},
    'Qle': {'rename_from': 'LE_F_MDS', 'unit': 'W/m²', 'description': 'Latent heat flux', 'missing_value': -9999},
    'Qh': {'rename_from': 'H_F_MDS', 'unit': 'W/m²', 'description': 'Sensible heat flux', 'missing_value': -9999},
    'Qg': {'rename_from': 'G_F_MDS', 'unit': 'W/m²', 'description': 'Ground heat flux', 'missing_value': -9999},
    'Qle_cor': {'rename_from': 'LE_CORR', 'unit': 'W/m²', 'description': 'Energy-balance-corrected latent heat flux', 'missing_value': -9999},
    'Qh_cor': {'rename_from': 'H_CORR', 'unit': 'W/m²', 'description': 'Energy-balance-corrected sensible heat flux', 'missing_value': -9999},
    'Qle_cor_uc': {'rename_from': 'LE_CORR_JOINTUNC', 'unit': 'W/m²', 'description': 'Qle_cor joint uncertainty', 'missing_value': -9999},
    'Qh_cor_uc': {'rename_from': 'H_CORR_JOINTUNC', 'unit': 'W/m²', 'description': 'Qh_cor joint uncertainty', 'missing_value': -9999},
    'Ustar': {'rename_from': 'USTAR','unit': 'm/s', 'description': 'Friction velocity', 'missing_value': -9999},
    'NEE': {'rename_from': 'NEE_VUT_REF', 'unit': 'umolCO2/m²/s', 'description': 'Net ecosystem exchange of CO2', 'missing_value': -9999},
    'NEE_uc': {'rename_from': 'NEE_VUT_REF_JOINTUNC', 'unit': 'umolCO2/m²/s', 'description': 'NEE joint uncertainty', 'missing_value': -9999},
    'GPP': {'rename_from': 'GPP_NT_VUT_REF', 'unit': 'umolCO2/m²/s', 'description': 'Gross primary productivity of CO2', 'missing_value': -9999},
    'GPP_se': {'rename_from': 'GPP_NT_VUT_SE', 'unit': 'umolCO2/m²/s', 'description': 'Standard error in GPP', 'missing_value': -9999},
    'GPP_DT': {'rename_from': 'GPP_DT_VUT_REF', 'unit': 'umolCO2/m²/s', 'description': 'Gross primary productivity of CO2 from daytime partitioning method', 'missing_value': -9999},
    'GPP_DT_se': {'rename_from': 'GPP_DT_VUT_SE', 'unit': 'umolCO2/m²/s', 'description': 'Standard error in GPP_DT', 'missing_value': -9999},
    'Resp': {'rename_from': 'RECO_NT_VUT_REF', 'unit': 'umolCO2/m²/s', 'description': 'Ecosystem respiration', 'missing_value': -9999},
    'Resp_se': {'rename_from': 'RECO_NT_VUT_SE', 'unit': 'umolCO2/m²/s', 'description': 'Standard error in Resp', 'missing_value': -9999}
}

# Extract selected variables and rename mapping
selected_variables = ['TIMESTAMP_START'] + [var_info['rename_from'] for var_info in variable_info.values() if 'rename_from' in var_info]
rename_mapping = {var_info['rename_from']: var_name for var_name, var_info in variable_info.items() if 'rename_from' in var_info}

# Function to extract station name from file path
def extract_station_name(file_path):
    station_name = os.path.basename(file_path).split('_')[1]  # Assuming station name is the second part after FLX_
    return station_name

# Function to extract start and end years from xarray dataset
def extract_start_end_years(xds):
    start_year = xds.time.dt.year.min().item()
    end_year = xds.time.dt.year.max().item()
    return start_year, end_year

# Process CSV files and save as NetCDF
for csv_file in csv_files:
    print(csv_file)
    df_insitu_flux = pd.read_csv(csv_file)
    
    # Convert DataFrame to xarray dataset
    insitu_data_nc = select_rename_convert_to_xarray(df_insitu_flux, selected_variables, rename_mapping)

    # Assign attributes to variables
    for var_name, var_info in variable_info.items():
        insitu_data_nc[var_name].attrs['unit'] = var_info.get('unit', '')
        insitu_data_nc[var_name].attrs['_FillValue'] = var_info.get('missing_value', 'None')
        insitu_data_nc[var_name].attrs['description'] = var_info.get('description', '')

    # Extract station name and start/end years
    station_name = extract_station_name(csv_file)
    start_year, end_year = extract_start_end_years(insitu_data_nc) 

    # Construct file name and path
    outputfile_name = f"fluxes_{station_name}_{start_year}_{end_year}.nc"
    outputfile_path = os.path.join(output_path, outputfile_name)
    
    # Save xarray dataset as NetCDF file
    insitu_data_nc.to_netcdf(outputfile_path)
    
    # Stop after processing one CSV file (for demonstration purposes)
    break
