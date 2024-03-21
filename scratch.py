import xarray as xr
import os
import matplotlib.pyplot as plt

# Set the working directory
os.chdir("/home/khanalp/code/PhD/preprocessICOSdata")

input_nc_location = "/home/khanalp/data/processed/input_pystemmus"
output_path = "/home/khanalp/code/PhD/preprocessICOSdata/output/plot"

# List all files in the directory
files_in_directory = os.listdir(input_nc_location)

# Filter files with .nc extension
nc_files = [file for file in files_in_directory if file.endswith(".nc")]

# Define the variables to plot
variables_to_plot = ['Tair', 'SWdown', 'LWdown', 'VPD', 'Psurf', 'Precip', 'Wind', 'RH', 'CO2air', 'LAI']

# Define the units and titles for each variable
variable_info = {
    'Tair': {'title': 'Temperature', 'unit': 'K'},
    'SWdown': {'title': 'Downward Shortwave Radiation', 'unit': 'W/sq.m'},
    'LWdown': {'title': 'Downward Longwave Radiation', 'unit': 'W/sq.m'},
    'VPD': {'title': 'Vapor Pressure Deficit', 'unit': 'hPa'},
    'Psurf': {'title': 'Surface Pressure', 'unit': 'Pa'},
    'Precip': {'title': 'Precipitation', 'unit': 'mm/s'},
    'Wind': {'title': 'Wind Speed', 'unit': 'm/s'},
    'RH': {'title': 'Relative Humidity', 'unit': '%'},
    'CO2air': {'title': 'CO2 Concentration', 'unit': 'ppm'},
    'LAI': {'title': 'Leaf Area Index', 'unit': 'sq.m/sq.m'}
}

# Calculate the number of rows and columns for subplots
num_plots = len(variables_to_plot)
num_rows = num_plots
num_cols = 1

for file in nc_files:
    
    #read dataset
    dataset = xr.open_dataset(os.path.join(input_nc_location,file))

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 2.5 * num_rows))

    # Flatten the axes array for easier iteration
    axes_flat = axes.flatten()

    # Iterate over variables and plot them
    for i, var in enumerate(variables_to_plot):
        # Get the subplot for the current variable
        ax = axes_flat[i]
        
        # Plot the variable
        ax.plot(dataset.time.values.flatten(), dataset[var].values.flatten(),
                linestyle='-',
                linewidth=1.8,
                color='black',
                label=var)
        
        # Set the title, xlabel, ylabel, and grid
        ax.set_title(f"{variable_info[var]['title']}")
        ax.set_xlabel("Date")
        # Set the ylabel concatenating title and unit
        ax.set_ylabel(f"{var} ({variable_info[var]['unit']})")
        ax.grid(linestyle='dotted')
        
        
    # Adjust the space between subplots
    plt.subplots_adjust(hspace=0.75) 
    
    
    # Split the filename by underscore
    parts = file.split('_')

    # Extract station name from the first part after FLX_
    station_name = parts[1]

    # Extract start_year and end_year from the last part
    years_part = parts[-1]
    start_year, end_year = years_part.split('-')
    
    # Construct the output filename
    output_filename = f"{station_name}_{start_year}_{end_year}.png"
    output_file = os.path.join(output_path,"plot",output_filename)
    # Save the plot in 300 dpi resolution
    plt.savefig('plot.png', dpi=300)
    
    # Show the plot
    plt.show()
    break
    