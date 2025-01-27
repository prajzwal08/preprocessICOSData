import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append("/home/khanalp/code/PhD")

from utils.unit_conversion import convert_jcm2_to_wm2

# Path to the data file
path_to_data = "/home/khanalp/data/fluxsites_NL/incoming/cabauw/uurgeg_348_2021-2030.txt"
# Save the resampled DataFrame to a CSV file
output_path = "/home/khanalp/data/fluxsites_NL/incoming/cabauw/intermediate/df_resampled.csv"

#   Read the data file
with open(path_to_data, 'r') as file:
    data = file.read()

# Split the data into lines
lines = data.split('\n')

# Initialize a dictionary to store the data
data_dict = {'YYYYMMDD': [], 
             'HH': [],
             'FH': [] , # Mean wind speed (in 0.1 m/s) for the hour preceding the observation time stamp
             'TA': [], # Temperature (in 0.1 degrees Celsius) at 1.50 m at the time of observation
             'TD': [], # Dew point temperature (in 0.1 degrees Celsius) at 1.50 m at the time of observation
             'Q': [], # Global radiation (in J/cm2) during the hourly division
             'RH': [], # Hourly precipitation amount (in 0.1 mm) (-1 for <0.05 mm)
             'P': [],  # Air pressure (in 0.1 hPa) reduced to mean sea level, at the time of observation
             'U': [] } # Relative atmospheric humidity (in percents) at 1.50 m at the time of observation

# Define the columns needed
columns_needed = ['STN,YYYYMMDD,', 'HH,','FH,', 'T,', 'TD,', 'Q,', 'RH,','P,', 'U,']

# Extract the data
columns = None # Initialize the columns
shift = -1 # To determine the column position in the data file 
conversion_factor = 0.1 # Needed for ‘FH’ : 0.1, ’T’ : 0.1, ‘TD’ : 0.1 , 'RH' : 0.1, 'P' : 0.1


for line in lines:
    if columns is None:
        if '#' in line:
            columns = line.split()
            indices_needed = [columns.index(col) + shift for col in columns_needed if col in columns]
    else:
        columns = line.split()
        if indices_needed and all(index < len(columns) for index in indices_needed):
        #    print(columns[indices_needed[0]].split(',')[1])
            data_dict['YYYYMMDD'].append(columns[indices_needed[0]].split(',')[1])
            data_dict['HH'].append(pd.to_numeric(columns[indices_needed[1]].split(',')[0]))
            data_dict['FH'].append(pd.to_numeric(columns[indices_needed[2]].split(',')[0]) * conversion_factor)
            data_dict['TA'].append(pd.to_numeric(columns[indices_needed[3]].split(',')[0]) * conversion_factor)
            data_dict['TD'].append(pd.to_numeric(columns[indices_needed[4]].split(',')[0]) * conversion_factor)
            data_dict['Q'].append(pd.to_numeric(columns[indices_needed[5]].split(',')[0]))
            data_dict['RH'].append(pd.to_numeric(columns[indices_needed[6]].split(',')[0]) * conversion_factor)
            data_dict['P'].append(pd.to_numeric(columns[indices_needed[7]].split(',')[0]) * conversion_factor)
            data_dict['U'].append(pd.to_numeric(columns[indices_needed[8]].split(',')[0]))

# Create a DataFrame from the dictionary
df = pd.DataFrame(data_dict)

# Convert HH to two-digit strings and handle the case where HH is 24
# Because in the data from KNMI 24 should actually be 00. ie. 2021-01-01 24 should be 2021-01-02 00
df['HH'] = df['HH'].apply(lambda x: '00' if x == 24 else str(x).zfill(2)) 
df['YYYYMMDD'] = df.apply(lambda row: (pd.to_datetime(row['YYYYMMDD'], format='%Y%m%d') + pd.Timedelta(days=1)).strftime('%Y%m%d') if row['HH'] == '00' else row['YYYYMMDD'], axis=1)

# Create datetime column and set it as index
df['datetime'] = pd.to_datetime(df['YYYYMMDD'] + df['HH'], format='%Y%m%d%H')
df.set_index('datetime', inplace=True)
df.drop(columns=['YYYYMMDD', 'HH'], inplace=True)

# Convert the 'Q' column from J/cm2 to W/m2
df['Q'] = convert_jcm2_to_wm2(energy_jcm2=df['Q'].values, duration_seconds=3600)

# idea on resampling is resample other variable linearly while for accumulated precipitation, simply divide into two halfs.
# Resample and interpolate variables
variables_to_linearly_interpolate = df.columns.difference(['RH']) # RH is the accumulated precipitation in hour (confusing notation from KNM)
df_resampled = df[variables_to_linearly_interpolate].resample('30T').asfreq().interpolate(method='linear')

# Resample and uniformly distribute 'RH'
df_resampled['RH'] = df['RH'].resample('30T').bfill()
df_resampled['RH'] /= 2  # Divide 'RH' by 2

# Copy the values from 01:00:00 to 00:30:00 for first columns (manual)
df_resampled.loc[pd.Timestamp('2021-01-01 00:30:00')] = df.loc[pd.Timestamp('2021-01-01 01:00:00')]

# Sort the dataframe by the datetime index
df_resampled.sort_index(inplace=True)

# Rename columns
df_resampled.rename(columns={
    'FH': 'Wind',
    'TA': 'Tair',
    'TD': 'Tdew',
    'Q': 'SWdown',
    'RH': 'Precip',
    'P': 'Psurf',
    'U': 'Rhumidity'
}, inplace=True)

# Add units as attributes
df_resampled.attrs['Wind'] = 'm/s'
df_resampled.attrs['Tair'] = 'degreeC'
df_resampled.attrs['Tdew'] = 'degreeC'
df_resampled.attrs['SWdown'] = 'W/sq.m'
df_resampled.attrs['Precip'] = 'mm'
df_resampled.attrs['Psurf'] = 'hPa'
df_resampled.attrs['Rhumidity'] = '%'

# Save the df_resampled to a CSV file
df_resampled.to_csv(output_path)

# just to see visually how the data looks like 
variables = df_resampled.columns
# Set the style for the plots
sns.set(style="whitegrid")

# Create a figure and axes for each variable
fig, axes = plt.subplots(nrows=len(variables), ncols=1, figsize=(12, 18), sharex=True)

# Plot each variable
for i, var in enumerate(variables):
    df_resampled[var].plot(ax=axes[i])
    axes[i].set_ylabel(var)
    axes[i].set_title(f'Time Series of {var}')

# Set the x-axis label
axes[-1].set_xlabel('Datetime')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
plt.savefig("/home/khanalp/data/fluxsites_NL/incoming/cabauw/intermediate/resampled_data.png")





