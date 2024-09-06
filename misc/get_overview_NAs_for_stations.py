"""
Author: [Prajwal Khanal]
Date: [21 March 1:58PM @ Boston Public Library ]
Purpose: In the code "calculateNAs.py", i calculate percentage of missing values for each variables and stations
in the ICOS data. Here, I want to see in which year and month the data are missing for each station, especially for five main variables (without 
VPD and CO2 air). 
If, there are less than 10 percent data gaps in each year-month combination, i plan to fill them statistically, 
while if all the values are missing, i need to find other steps.

Result: "overview_missingdata_for_station.csv" finds that for 33 stations three variables, Psurf, P and Wind are 
are missing systematically in 2020. So, for now, I will run the model until 2019 for them.
"""

import os
import pandas as pd

def load_missing_data_info(csv_files, missing_data_info_path):
    """
    Load missing data information from CSV files, preprocess, and filter.
    
    Args:
        csv_files (list): List of CSV files containing missing data information.
        missing_data_info_path (str): Path to directory containing CSV files.
    
    Returns:
        list: List of preprocessed DataFrames for each station.
    """
    dfs = []
    # Filter missing data based on threshold. So, if any year-month has greater than 10 percent missing, it will be considered. 
    threshold = 10
    excluded_columns = ['VPD','CO2air']
    
    for file in csv_files:
        station_name = file.split('.')[0]
        file_path = os.path.join(missing_data_info_path, file)
        data_frame = pd.read_csv(file_path)
        data_frame = data_frame.rename(columns={'date': 'year', 'date.1': 'month'})
        data_frame_excluded = data_frame.drop(columns=excluded_columns)
        
      
        filtered_data = data_frame_excluded.iloc[:, 2:][data_frame_excluded.iloc[:, 2:] > threshold]
        df_filtered = pd.concat([data_frame_excluded[['year', 'month']], filtered_data], axis=1)
        df_column_removed = df_filtered.dropna(axis=1, how='all')
        df_only_missing = df_column_removed.dropna(subset=df_column_removed.columns[2:], how='all')
        
        # Check if the DataFrame is empty before adding it to the list
        if not df_only_missing.empty:
            df_only_missing['station_name'] = station_name
            dfs.append(df_only_missing)
    
    return dfs

def combine_dataframes(dfs):
    """
    Combine DataFrames into a single DataFrame with station names included.
    
    Args:
        dfs (list): List of preprocessed DataFrames for each station.
    
    Returns:
        DataFrame: Combined DataFrame containing all the data with station names.
    """
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

# Main script
if __name__ == "__main__":
    missing_data_info_path = "/home/khanalp/code/PhD/preprocessICOSdata/output/csvs/countNAs"
    csv_files = [file for file in os.listdir(missing_data_info_path) if file.endswith('.csv')]
    
    # Load missing data information
    dfs = load_missing_data_info(csv_files, missing_data_info_path)
    
    # Combine DataFrames
    combined_df = combine_dataframes(dfs)
    
    # Output combined DataFrame
    print(combined_df)
