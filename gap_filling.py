import os
import pandas as pd
import numpy as np
from typing import Union
from unit_conversion import convert_local_to_utc

insitu_data_path =  "~/data/fluxsites_NL/incoming/veenkampen/intermediate/cleaned_insitu_data_for_gap_filling.csv"
era5land_data_path =  "~/data/fluxsites_NL/incoming/veenkampen/intermediate/cleaned_era5land_data_for_gap_filling.csv"

df_insitu = pd.read_csv(insitu_data_path, index_col=0)
df_era5land = pd.read_csv(era5land_data_path, index_col=0)

df_insitu['time'] = pd.to_datetime(df_insitu['time'])
df_era5land['time'] = pd.to_datetime(df_era5land['time'])

instantaneous_variable_era5land = ['air_pressure','air_temperature','VPD','wind_speed']
accumulative_variable_era5land = ['precipitation','swdown','lwdown']

df_insitu_accumulative_variable = df_insitu[accumulative_variable_era5land + ["time"]]

time_difference = +1 # wrt UTC
df_insitu['utc_time'] = convert_local_to_utc(df_insitu['time'].values,time_difference)
df_insitu_accumulative_variable['utc_time'] = convert_local_to_utc(
    df_insitu_accumulative_variable['time'].values,time_difference)

# Assuming df_insitu_accumulative_variable is your DataFrame
df = df_insitu_accumulative_variable.copy().iloc[2:,:].drop('time',axis =1 )

# Convert 'utc_time' to datetime if not already done
df['utc_time'] = pd.to_datetime(df['utc_time'])

# Set 'utc_time' as the index
df.set_index('utc_time', inplace=True)

resampled_df = df.resample('1H',closed='right', label='right').sum()

resampled_df["2015-01-01"]

filtered_df = resampled_df.loc['2015-01-01']

filtered_df['cumulative_precipitation'] = filtered_df['precipitation'].groupby(filtered_df.index.date).cumsum()
filtered_df['cumulative_swdown'] = filtered_df['swdown'].groupby(filtered_df.index.date).cumsum()
filtered_df['cumulative_lwdown'] = filtered_df['lwdown'].groupby(filtered_df.index.date).cumsum()



# Reset the index if needed
resampled_df.reset_index(inplace=True)

print(resampled_df)
                                    
# Replace all negative values in the 'swdown' column with 0
df_insitu_accumulative_variable.loc[df_insitu_accumulative_variable['swdown'] < 0, 'swdown'] = 0    
                                    




 # Merge the filtered df_insitu with df_era5land
merged_df = pd.merge(
    df_era5land,
    filtered_df_insitu,
    left_on='time',
    right_on='utc_time',
    suffixes=('_era5land', '_insitu')
)

selected_columns = [f'{var}_era5land' for var in instantaneous_variable_era5land] + [f'{var}_insitu' for var in instantaneous_variable_era5land] + ['time_era5land']

# Create DataFrame with the selected columns
result_df = merged_df[selected_columns]







