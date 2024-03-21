import numpy as np
import pandas as pd
import xarray as xr

def get_co2_data(longitude, latitude, start_time, end_time, file_path, resampling_interval = '30min'):
    """
    Retrieve CAMS CO2 data for a specific location and time period.
    
    Parameters:
    longitude (float): Longitude of the location.
    latitude (float): Latitude of the location.
    start_time (str): Start time in the format 'YYYY-MM-DD'.
    end_time (str): End time in the format 'YYYY-MM-DD'.
    file_path (str): Path to the CAMS netCDF file.
    
    Returns:
    co2_data (np.ndarray): Array of CO2 data in parts per million (ppm). Since co2 data for cams is in kg/kg.
    """
     # Constants
    M_CO2 = 44.01  # Molar mass of CO2 in g/mol
    M_dry_air = 28.97  # Molar mass of dry air in g/mol
    
     # Conversion factor from kg/kg to ppm
    conversion_factor = 1e6  # ppm
    
    # Open the CAMS dataset
    cams = xr.open_dataset(file_path).sortby('time')
    
    # Select the nearest location to the provided longitude and latitude
    cams_location_selected = cams.sel(latitude=latitude, longitude=longitude, method='nearest')
    
    # Select the time range
    cams_date_selected = cams_location_selected.sel(time=slice(start_time, end_time))
    
    # Extract time and CO2 variables
    time_data = cams_date_selected['time'].values
    co2_data = cams_date_selected['co2'].values.reshape(-1)
    
    # Create a pandas DataFrame
    df = pd.DataFrame({'time': time_data, 'co2': co2_data})
    
    # Set 'time' column as index
    df.set_index('time', inplace=True)
    
    # Resample to 30-minute intervals and forward fill missing values
    df_filled = df.resample(resampling_interval).ffill()
    
    # Extend the index until 'end_time' and forward fill
    end_date_extend = pd.to_datetime(end_time)
    df_filled_new = df_filled.reindex(pd.date_range(start=df_filled.index.min(), end=end_date_extend, freq='30min')).ffill()
    
    # Extract CO2 data as numpy array
    co2 = np.array(df_filled_new['co2'])
    
    # Convert kg/kg to ppm
    co2_ppm = co2 * (conversion_factor * M_dry_air / M_CO2)
    
    return co2_ppm