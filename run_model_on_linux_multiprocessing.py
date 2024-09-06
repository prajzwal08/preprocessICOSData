"""
Author: Prajwal Khanal
Date: April 23, 2024 @ 10:23 @ Hengelo 
Purpose: Run model simulations for multiple sites in parallel (with chunk size of 20)
"""

import os
import sys
import pandas as pd
from PyStemmusScope import StemmusScope
from PyStemmusScope import save
from datetime import datetime
import multiprocessing


def run_model_for_site(args):
    """
    Run the model simulation for a specific site.

    Args:
        site_name (str): Name of the site.
        start_time (str): Start time of the simulation.
        end_time (str): End time of the simulation.
    """
    site_name, start_time, end_time = args
    
    # Define path to the log directory
    log_dir = "/home/khanalp/logs/pystemmusscoperun/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Define the path to the log file for this site
    log_file = os.path.join(log_dir, f"{site_name}_log.txt")
    
    
    # Redirect standard output and standard error to the log file
    sys.stdout = open(log_file, 'w')
    sys.stderr = open(log_file, 'w')
    
    # Define the path to the workspace
    workspace = "/home/khanalp/STEMMUSSCOPE/STEMMUS_SCOPE"
    
    # Define the path to the configuration file for the site
    config_file_path = os.path.join(workspace, 'ICOS_sites', site_name, 'config_file.txt')

    # Define the path to the executable file
    exe_file_path = os.path.join(workspace, 'STEMMUS_SCOPE', 'run_model_on_snellius', 'exe', 'STEMMUS_SCOPE')    

    # Define the path to MATLAB
    matlab_path = '/opt/matlab' 

    # Set LD_LIBRARY_PATH
    os.environ['LD_LIBRARY_PATH'] = (
         f"{matlab_path}/MATLAB_Runtime/v910/runtime/glnxa64:"
         f"{matlab_path}/MATLAB_Runtime/v910/bin/glnxa64:"
         f"{matlab_path}/MATLAB_Runtime/v910/sys/os/glnxa64:"
         f"{matlab_path}/MATLAB_Runtime/v910/extern/bin/glnxa64:"
         f"{matlab_path}/MATLAB_Runtime/v910/sys/opengl/lib/glnxa64")

    # Create an instance of the model
    model = StemmusScope(config_file=config_file_path, model_src_path=exe_file_path)

    # Setup the model
    config_path = model.setup(
        Location=site_name,
        StartTime=start_time,
        EndTime=end_time)

    # Print the new config file path
    print(f"New config file {config_path}")

    # Print the model input and output directories
    print(f'Model input dir {model.config["InputPath"]}')
    print(f'Model output dir {model.config["OutputPath"]}')

    # Start the timer
    start_timer = datetime.now()

    # Logger:
    print(f"Running model for site: {site_name}")

    # Run the model
    result = model.run()

    # End the timer
    end_timer = datetime.now()

    # Calculate the duration
    duration = end_timer - start_timer
    print(f"Model run time: {duration}")

    # Save output in netcdf format
    required_netcdf_variables = os.path.join(workspace, 'STEMMUS_SCOPE', 'utils', 'csv_to_nc', 'required_netcdf_variables.csv')
    nc_file_name = save.to_netcdf(config_path, required_netcdf_variables)
    print(nc_file_name)    
    # Example: Print completion message
    print(f"Model run for site {site_name} completed.")

def run_model_in_chunks(chunk):
    # creating a pool object 
    p = multiprocessing.Pool(processes=20)
    # map list to target function 
    p.map(run_model_for_site, chunk)
    p.close()
    p.join()

if __name__ == "__main__":
    
    path_station_info = "/home/khanalp/code/PhD/preprocessICOSdata/output/csvs/stations_readyformodelrun.csv"
    data_frame = pd.read_csv(path_station_info, index_col = 0)
    
    # Assuming df is your DataFrame containing columns 'Station_Name', 'Start_Time', and 'End_Time'
    # Extract the first 15 rows from the DataFrame
    first_15_rows = data_frame.head(15) # Already model run for this. 
    
    # Extract everything except the first 15 rows
    remaining_rows = data_frame.iloc[15:]

    # Extract "DE-Kli" from the first 15 rows
    specific_row = data_frame[data_frame['Station_Name'] == 'DE-Kli'].head(1)

    # Concatenate remaining_rows with specific_row
    result = pd.concat([specific_row, remaining_rows])

    # Create a list of tuples from the extracted rows
    mylist = []

    for index, row in result.iterrows():
       
    #    #------------------Just for trial run------------#
    #     #Parse the date from Start_Time 
    #     start_date = pd.to_datetime(row['Start_Time']).date()
    #     # Construct End_Time with the same date and time 23:30
    #     end_time = pd.Timestamp(start_date.strftime('%Y-%m-%d') + ' 23:30')
    #     # Convert End_Time to the same format as Start_Time
    #     end_time_formatted = end_time.strftime('%Y-%m-%dT%H:%M')
    #     #----------------Just for trial ended. 
        
        # Append tuple to the list
        mylist.append((row['Station_Name'], row['Start_Time'], row['End_Time']))
    
    # Split the list into chunks of 20
    chunk_size = 20
    chunks = [mylist[i:i+chunk_size] for i in range(0, len(mylist), chunk_size)]
    
    # Run each chunk in a separate multiprocessing pool
    for chunk in chunks:
        run_model_in_chunks(chunk)
    

