"""
Author: Prajwal Khanal 
Date: 2024-04-10 @ 10 AM @ Hengelo
Purpose: This program generates comparison plots of LAI data from MODIS and Copernicus satellites for different flux stations.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Define file paths
file_path_stationinfo = "/home/khanalp/code/PhD/preprocessICOSdata/output/csvs/stationdetails.csv"
file_path_modis_lai = "/home/khanalp/data/processed/lai/modis/"
file_path_copernicus_lai = "/home/khanalp/data/processed/lai/copernicus/"
output_file_path = "/home/khanalp/code/PhD/preprocessICOSdata/output/plot/laicomparison/"

# Read station information
station_info = pd.read_csv(file_path_stationinfo)
# Replace spaces in column names with underscores
station_info.columns = station_info.columns.str.replace(' ', '_')


# Counter to track the number of loops (for trials only)
# loop_count = 0

# Loop through each station
for index, row in station_info.iterrows():
    
    # Extract station name and IGBP short name 
    station_name = row['station_name']
    igbp_short_name = row['IGBP_short_name']

    # Print start message for the station
    print(f"Processing data for station: {station_name}...")
    
    # Find paths to MODIS LAI and Copernicus LAI files for the station
    modis_lai_path = [os.path.join(root, file) for root, dirs, files in os.walk(file_path_modis_lai) for file in files if file.endswith('.csv') and station_name in file]
    copernicus_lai_path = [os.path.join(root, file) for root, dirs, files in os.walk(file_path_copernicus_lai) for file in files if file.endswith('.csv') and station_name in file]
    
    # Read MODIS LAI data
    df_modis_lai = pd.read_csv(modis_lai_path[0], index_col=0)
    df_modis_lai = df_modis_lai.loc[df_modis_lai.index <= '2019-12-31 23:30:00']
    df_modis_lai.index = pd.to_datetime(df_modis_lai.index)
    
    # Read Copernicus LAI data
    df_copernicus_lai = pd.read_csv(copernicus_lai_path[0], index_col=0)
    df_copernicus_lai.index = pd.to_datetime(df_copernicus_lai.index)
    
    # Set font family and tick label sizes
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='small')  # Adjust the size as needed
    plt.rc('ytick', labelsize='small')  # Adjust the size as needed

    # Create a figure and axis
    fig = plt.figure(figsize=(10, 5))  # Adjust figsize for a longer x-axis
    ax = fig.add_subplot(1, 1, 1)

    # Plot MODIS LAI data
    modis_plot, = ax.plot(df_modis_lai.index, df_modis_lai.LAI, color='k', ls='solid', label='MODIS LAI')

    # Plot Copernicus LAI data
    copernicus_plot, = ax.plot(df_copernicus_lai.index, df_copernicus_lai.LAI, color='0.50', ls='dashed', label='Copernicus LAI')

    # Set labels for x and y axes
    ax.set_xlabel('Date', fontsize='medium')  # Adjust the font size as needed
    ax.set_ylabel('LAI (sq.m/sq.m)', fontsize='medium')  # Adjust the font size as needed
    
    # Add legend with specified labels and adjust font size
    ax.legend(handles=[modis_plot, copernicus_plot], loc='upper right', fontsize='medium', bbox_to_anchor=(0.99, 0.99), borderaxespad=0)

    # Set the title with the dynamic station name and IGBP short name
    ax.set_title(f"Comparison of LAI for the flux station {station_name} (Landuse: {igbp_short_name})", fontsize='large') 

    # Adjust layout for better appearance
    plt.tight_layout()

    # Save the plot with the station name as the filename
    output_file = os.path.join(output_file_path, f"{station_name}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.close()  # Close the current plot before moving to the next station
    
    # Print end message for the station
    print(f"Processing for station {station_name} completed.\n")
    # # Increment loop counter
    # loop_count += 1
    # if loop_count == 2:
    #     break  # Exit the loop after two iterations
