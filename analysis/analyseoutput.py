import xarray as xr
import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import gaussian_kde

from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import linregress

from utils import list_folders_with_prefix, list_csv_files_in_folder,read_csv_file_with_station_name
from utils import select_rename_convert_to_xarray

file_path_station_info = "/home/khanalp/code/PhD/preprocessICOSdata/output/csvs/stations_readyformodelrun.csv"
input_data_path_flux = "/home/khanalp/data/ICOS2020"
prefix = "FLX"
pystemmus_output_model_path = "/home/khanalp/STEMMUSSCOPE/STEMMUS_SCOPE/ICOS_sites/"

# List folders and CSV files 
folders = list_folders_with_prefix(input_data_path_flux, prefix)
csv_files = []
for folder in folders:
    folder_path = os.path.join(input_data_path_flux, folder)
    csv_files.extend(list_csv_files_in_folder(folder_path, "FULLSET_HH"))
    
station_info = pd.read_csv(file_path_station_info, index_col = 0)

# Assuming "Station_name" is the column containing station names
stations_to_plot = station_info.head(15)

selected_variables = [
    'TIMESTAMP_START',
    'NETRAD',
    #'SW_OUT',
    'G_F_MDS',
    'LE_CORR',
    'H_CORR',
    'LE_CORR_JOINTUNC',
    'H_CORR_JOINTUNC',
    'USTAR',
    'NEE_VUT_REF',
    'NEE_VUT_REF_JOINTUNC',
    'GPP_NT_VUT_REF',
    'GPP_NT_VUT_SE',
    'GPP_DT_VUT_REF',
    'GPP_DT_VUT_SE',
    'RECO_NT_VUT_REF',
    'RECO_NT_VUT_SE'
]

#Renaming them 
rename_mapping = {'NETRAD':'Rnet',
          #'SW_OUT':'SWup',
          'G_F_MDS':'Qg',
          'LE_CORR':'Qle',
          'H_CORR':'Qh',
          'LE_CORR_JOINTUNC':'Qle_cor_uc',
          'H_CORR_JOINTUNC':'Qh_cor_uc',
          'USTAR':'Ustar',
          'NEE_VUT_REF':'NEE',
          'NEE_VUT_REF_JOINTUNC' : 'NEE_uc',
          'GPP_NT_VUT_REF':'GPP',
          'GPP_NT_VUT_SE':'GPP_se',
          'GPP_DT_VUT_REF':'GPP_DT',
          'GPP_DT_VUT_SE':'GPP_DT_se',
          'RECO_NT_VUT_REF':'Resp',
          'RECO_NT_VUT_SE':'Resp_se'
         }

for index,row in stations_to_plot.iterrows():
    station = row['Station_Name']
    #igpb = row['IGBP short name']
    ## Reading model output
    modeloutput_folder_path = os.path.join(pystemmus_output_model_path, station,"output")
    # Get the latest folder created
    latest_folder = max((entry.path for entry in os.scandir(modeloutput_folder_path) if entry.is_dir()), key=os.path.getctime)
    # Filter out NetCDF files
    nc_files = [file for file in os.listdir(latest_folder) if file.endswith('.nc')]
    
    try: 
        model_output_nc = xr.open_dataset(os.path.join(latest_folder,nc_files[0]))
    except KeyError: 
         continue
        
    ## Reading insitu data 
    filtered_files = [file for file in csv_files if station in file]
    df_insitu_flux = pd.read_csv(filtered_files[0])
    #selecting required variables from ICOS data for input.
    # Check if 'NETRAD' is in the columns, if not, remove it from selected_variables
    if 'NETRAD' not in df_insitu_flux.columns:
        selected_variables = [var for var in selected_variables if var != 'NETRAD']
        
    insitu_data_nc_complete =  select_rename_convert_to_xarray(df_insitu_flux,selected_variables,rename_mapping)
    # Replace -9999 with NaN
    insitu_data_nc_complete = insitu_data_nc_complete.where(insitu_data_nc_complete != -9999, np.nan)
        # Find the maximum time of model_output_nc
    max_model_time = model_output_nc.time.max().values

    # Truncate insitu_data_nc to match the maximum time of model_output_nc
    insitu_data_nc = insitu_data_nc_complete.sel(time=slice(None, max_model_time))
    
    # Assuming 'insitu_data_filled' is your xarray Dataset
    insitu_data_nc['GPP'] = xr.DataArray((insitu_data_nc['GPP'].values.flatten() * 1e-6 * 12.01 * 1e-3).reshape(1,1,-1),dims=['x','y','time'])
    insitu_data_nc['NEE'] = xr.DataArray((insitu_data_nc['NEE'].values.flatten() * 1e-6 * 12.01 * 1e-3).reshape(1,1,-1),dims=['x','y','time'])
    
    variables_to_plot = ['Rnet','Qg','Qh','Qle','GPP','NEE']
    
    # Define the units and titles for each variable
    variable_info = {
        'Rnet': {'title': 'Net Radiation', 'unit': 'W/sq.m'},
        'Qg': {'title': 'Ground Heat Flux', 'unit': 'W/sq.m'},
        'Qh': {'title': 'Sensible Heat Flux', 'unit': 'W/sq.m'},
        'Qle': {'title': 'Latent Heat Flux', 'unit': 'W/sq.m'},
        'GPP': {'title': 'Gross Primary Productivity', 'unit': 'kg/sq.m/s'},
        'NEE': {'title': 'Net Ecosystem Exchange', 'unit': 'kg/sq.m/s'}
    }
    
    plt.clf()
    # Calculate the number of rows and columns for subplots
    num_plots_line = len(variables_to_plot)
    num_rows_line = num_plots_line
    num_cols_line = 1
    
        # Create subplots
    fig, axes = plt.subplots(num_rows_line, num_cols_line, figsize=(20, 2.5 * num_rows_line))

    # Flatten the axes array for easier iteration
    axes_flat = axes.flatten()
            
    # Plot the variable
    for i,var in enumerate(variables_to_plot):
        # Get the subplot for the current variable
        ax = axes_flat[i]
        try:
            # Plot insitu data
            ax.plot(insitu_data_nc.time.values.flatten(), insitu_data_nc[var].values.flatten(),
                    linestyle='-',
                    linewidth=1.8,
                    color='red',
                    label='in-situ', 
                    alpha = 0.9 )
        except KeyError:
            print(f"Insitu data for variable '{var}' not available.")
        try:
            # Plot model data
            ax.plot(model_output_nc.time.values.flatten(), model_output_nc[var].values.flatten(),
                    linestyle='-',
                    linewidth=1.8,
                    color='blue',
                    label='model')
        except KeyError:
            print(f"Model data for variable '{var}' not available.")
    
        # Set the title, xlabel, ylabel, and grid
        ax.set_title(f"{variable_info[var]['title']}")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{var} ({variable_info[var]['unit']})")
        ax.grid(linestyle='dotted')

        # Add legend
        ax.legend()
            
        # Adjust the space between subplots
    plt.suptitle(f"Comparison between PyStemmusscope and In-situ data for {station}")
    plt.subplots_adjust(hspace=0.75) 
    # plt.suptitle(f"Comparison between PyStemmusscope and In-situ data for {station} ({igpb})")
    # Save the plot with the station name as the filename
    plt.savefig(os.path.join("/home/khanalp/code/PhD/preprocessICOSdata/output/plot/comparisonts/", f"{station}_comparison.png"), dpi=300)
    # Show the plot
    