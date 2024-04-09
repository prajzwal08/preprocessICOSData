"""
Author: Prajwal Khanal
Date: April 9, 2024 @ 10:23 @ Hengelo
Purpose: Run model simulations for multiple sites in parallel
"""

import os
import sys
from PyStemmusScope import StemmusScope
from PyStemmusScope import save
from datetime import datetime

def run_model_for_site(site_name, start_time, end_time):
    """
    Run the model simulation for a specific site.

    Args:
        site_name (str): Name of the site.
        start_time (str): Start time of the simulation.
        end_time (str): End time of the simulation.
    """

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

if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 4:
        print("Usage: python3 run_model_linux.py <site_name> <start_time> <end_time>")
        sys.exit(1)
    
    # Extract arguments
    site_name = sys.argv[1]
    start_time = sys.argv[2]
    end_time = sys.argv[3]
    print(site_name,start_time,end_time)
    
    # Run the model for the specified site
    run_model_for_site(site_name, start_time, end_time)
