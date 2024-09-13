"""
Purpose: Run model simulations for single sites 
"""

import os
import sys
import pandas as pd
from PyStemmusScope import StemmusScope
from PyStemmusScope import save
from datetime import datetime

info_config_file_path = "/home/khanalp/code/PhD/preprocessICOSdata/csvs/04_info_for_configfile.csv"
df_station_details = pd.read_csv(info_config_file_path, index_col=0)

station_name = "NL-Cab"
# Filter DataFrame for the specific station name
station_details = df_station_details[df_station_details['Station_Name'] == station_name]
start_time = station_details['Start_Time'].iloc[0]
end_time = station_details['End_Time'].iloc[0]

# Define path to the log directory
# log_dir = "/home/khanalp/logs/pystemmusscoperun/"
# os.makedirs(log_dir, exist_ok=True)

# Define the path to the log file for this site
# log_file = os.path.join(log_dir, f"{station_name}_log.txt")

# Redirect standard output and standard error to the log file
# sys.stdout = open(log_file, 'w')
# sys.stderr = open(log_file, 'w')

# Define the path to the workspace
workspace = '/home/khanalp/STEMMUS_SCOPE_model/STEMMUS_SCOPE_new/STEMMUS_SCOPE'

config_file_location = '/home/khanalp/code/PhD/preprocessICOSdata/configfiles'


# Define the path to the configuration file for the site
config_file_path = os.path.join(config_file_location, station_name, 'config_file.txt')

# Define the path to the executable file
exe_file_path = os.path.join(workspace, 'run_model_on_snellius', 'exe', 'STEMMUS_SCOPE')    

# These are for running old executable file. 
# Define the path to MATLAB
# matlab_path = '/opt/matlab' 

# # Set LD_LIBRARY_PATH
# os.environ['LD_LIBRARY_PATH'] = (
#         f"{matlab_path}/MATLAB_Runtime/v910/runtime/glnxa64:"
#         f"{matlab_path}/MATLAB_Runtime/v910/bin/glnxa64:"
#         f"{matlab_path}/MATLAB_Runtime/v910/sys/os/glnxa64:"
#         f"{matlab_path}/MATLAB_Runtime/v910/extern/bin/glnxa64:"
#         f"{matlab_path}/MATLAB_Runtime/v910/sys/opengl/lib/glnxa64")

matlab_version = 'R2023a' #choose 'R2023a' or 'v910'
matlab_path = '/opt/matlab/MATLAB_Runtime/' + matlab_version #!whereis MATLAB  
#matlab_path = matlab_path.s.split(": ")[1]
os.environ['LD_LIBRARY_PATH'] = (
     f"{matlab_path}/runtime/glnxa64:"
     f"{matlab_path}/bin/glnxa64:"
     f"{matlab_path}/sys/os/glnxa64:"
     f"{matlab_path}extern/bin/glnxa64:"
     f"{matlab_path}sys/opengl/lib/glnxa64")
print(os.environ['LD_LIBRARY_PATH'])

# Create an instance of the model
model = StemmusScope(config_file=config_file_path, model_src_path=exe_file_path)

# Setup the model
config_path = model.setup(
    Location=station_name,
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
print(f"Running model for site: {station_name}")

# Run the model
result = model.run()

# End the timer
end_timer = datetime.now()

# Calculate the duration
duration = end_timer - start_timer
print(f"Model run time: {duration}")

# Save output in netcdf format
required_netcdf_variables = os.path.join(workspace, 'utils', 'csv_to_nc', 'required_netcdf_variables.csv')
nc_file_name = save.to_netcdf(config_path, required_netcdf_variables)
print(nc_file_name)    
# Example: Print completion message
print(f"Model run for site {station_name} completed.")

