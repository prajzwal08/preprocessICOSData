"""
This script processes ICOS flux data stored in CSV files (FLUXNET format), converts it to xarray dataset,
assigns attributes to variables, and saves the dataset as NetCDF files.
"""

import os
import sys
import pandas as pd

# Add the utils folder to the system path to use custom functions. 
sys.path.append('/home/khanalp/code/PhD/') 
from utils.utils import read_csv_file_with_station_name, select_rename_convert_to_xarray

# Define paths
station_details = '/home/khanalp/code/PhD/preprocessICOSdata/csvs/02_station_with_elevation_heightcanopy.csv'  
ICOS_location = "/home/khanalp/data/ICOS2020"
output_path = "/home/khanalp/data/processed/insituflux_with_qc"

variable_info = {
    'Rnet': {'rename_from': 'NETRAD', 'unit': 'W/m²', 'description': 'Net radiation', 'missing_value': -9999},
    'Rnet_qc' :{'rename_from': 'NETRAD_QC', 'unit': '0 to 1, indicating percentage of measured data', 'description': 'Quality flag for net radiation', 'missing_value': -9999},
    'SWup': {'rename_from': 'SW_OUT', 'unit': 'W/m²', 'description': 'Upward shortwave radiation', 'missing_value': -9999},
    'Qle': {'rename_from': 'LE_F_MDS', 'unit': 'W/m²', 'description': 'Latent heat flux', 'missing_value': -9999},
    'Qle_qc': {'rename_from': 'LE_F_MDS_QC', 'unit': '0-measured,1-good quality gapfill, 2-medium, 3-poor', 'description': 'Quality flag for latent heat flux', 'missing_value': -9999},
    'Qh': {'rename_from': 'H_F_MDS', 'unit': 'W/m²', 'description': 'Sensible heat flux', 'missing_value': -9999},
    'Qh_qc': {'rename_from': 'H_F_MDS_QC', 'unit': '0-measured,1-good quality gapfill, 2-medium, 3-poor', 'description': 'Quality flag for sensible heat flux', 'missing_value': -9999},
    'Qg': {'rename_from': 'G_F_MDS', 'unit': 'W/m²', 'description': 'Ground heat flux', 'missing_value': -9999},
    'Qg_qc': {'rename_from': 'G_F_MDS_QC', 'unit': '0-measured,1-good quality gapfill, 2-medium, 3-poor', 'description': 'Quality flag for ground heat flux', 'missing_value': -9999},
    'Qle_cor': {'rename_from': 'LE_CORR', 'unit': 'W/m²', 'description': 'Energy-balance-corrected latent heat flux', 'missing_value': -9999},
    'Qh_cor': {'rename_from': 'H_CORR', 'unit': 'W/m²', 'description': 'Energy-balance-corrected sensible heat flux', 'missing_value': -9999},
    'Qle_cor_uc': {'rename_from': 'LE_CORR_JOINTUNC', 'unit': 'W/m²', 'description': 'Qle_cor joint uncertainty', 'missing_value': -9999},
    'Qh_cor_uc': {'rename_from': 'H_CORR_JOINTUNC', 'unit': 'W/m²', 'description': 'Qh_cor joint uncertainty', 'missing_value': -9999},
    'Ustar': {'rename_from': 'USTAR','unit': 'm/s', 'description': 'Friction velocity', 'missing_value': -9999},
    'NEE': {'rename_from': 'NEE_VUT_REF', 'unit': 'umolCO2/m²/s', 'description': 'Net ecosystem exchange of CO2', 'missing_value': -9999},
    'NEE_qc': {'rename_from': 'NEE_VUT_REF_QC', 'unit': '0-measured,1-good quality gapfill, 2-medium, 3-poor', 'description': 'Quality flag for net ecosystem exchange of CO2', 'missing_value': -9999},
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

# Main script execution
if __name__ == "__main__":
    df_station_details = pd.read_csv(station_details)
    # Process data for each station listed in the station details CSV
  
    for index, station_info in df_station_details.iterrows():
        station_name = str(station_info['station_name'])
         # Iterate over the filenames in the output path
        station_exists = False
        for filename in os.listdir(output_path):
            # Extract station name from filename
            if station_name == filename.split('_')[1]:
                print(f"Station '{station_name}' already exists.")
                station_exists = True
                break
        
        if not station_exists:
            print(f"Station '{station_name}' does not exist.")
            try:
            # Read the corresponding CSV file for the ICOS station in fluxnet format
                df_insitu = read_csv_file_with_station_name(ICOS_location, station_name)
                if df_insitu is not None:
                    # Convert DataFrame to xarray dataset
                    xr_insitu_flux = select_rename_convert_to_xarray(df_insitu, selected_variables, rename_mapping)
                     # Assign attributes to variables
                    for var_name, var_info in variable_info.items():
                        xr_insitu_flux[var_name].attrs['unit'] = var_info.get('unit', '')
                        xr_insitu_flux[var_name].attrs['_FillValue'] = var_info.get('missing_value', 'None')
                        xr_insitu_flux[var_name].attrs['description'] = var_info.get('description', '')
                        
                        start_year, end_year = xr_insitu_flux.time.dt.year.min().item(),xr_insitu_flux.time.dt.year.max().item()
                                        # Construct file name and path
                        outputfile_name = f"fluxes_{station_name}_{start_year}_{end_year}.nc"
                        outputfile_path = os.path.join(output_path, outputfile_name)
                        
                        # Save xarray dataset as NetCDF file
                        xr_insitu_flux.to_netcdf(outputfile_path)
                else:
                    print(f"No data found for station {station_name}. Skipping...")  
            except Exception as e:
                print(f"An error occurred while processing station {station_name}: {e}")

                            
            
            


