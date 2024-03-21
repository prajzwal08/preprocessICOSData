import pandas as pd
import numpy as np




def check_data_availability_LAI(filled_lai,lai_time, start_year ,end_year):
    """
    Checks the availability of MODIS LAI data and fills missing values if necessary. It ensures that the provided time range matches the expected range for MODIS observations.
    
    Parameters:
        filled_lai (numpy.ndarray): Array containing MODIS LAI data.
        lai_time (pandas.Series): Series containing timestamps corresponding to the MODIS LAI data.
    
    Returns:
        tuple: A tuple containing the filled LAI data and the corresponding dates.
    """
    
    # Check if filled_lai has no NAs or not, because sometimes MODIS observations are missing in between.
    all_tsteps = []
    
    # Loop over each year from start to end of MODIS data 
    for year in range(lai_time.dt.year.min(), lai_time.dt.year.max()+1):
        # Generate the first two dates for the year
        #year_dates = pd.date_range(start=f"{year}-01-01", periods=2, freq='8D')
        
        # Ensure 46 evenly spaced dates for the rest of the year
        year_dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='8D')
        
        # Append the datetime series for the current year to the list
        all_tsteps.extend(year_dates)
    
    #Change to pd datetime. 
    all_time = pd.to_datetime(all_tsteps)
    #Check if all date spaced 8 days apart are present in original MODIS lai_time. 
    result = all_time.isin(lai_time)
    
    temp_array = np.full(result.shape, np.nan)
    
    # Fill new array based on the condition of result.
    #This basically fills the point after checking date. 
    fill_index = 0
    for i, val in enumerate(result):
        if val:
            temp_array[i] = filled_lai[fill_index]
            fill_index += 1
    
    selected_lai = temp_array.reshape(-1, 46)[1:-1] #Basically because each year MODIS has 46 observations, if all are available.
    
    # Select dates from 2003 to 2022 (basically removing 2002 and 2023 because they are not complete.)
    selected_dates = all_time[(all_time.year >= start_year) & (all_time.year <= end_year)]
    
    if len(np.isnan(selected_lai).flatten()) > 0:
        selected_lai_flatten = selected_lai.reshape(-1)
        positions = np.where(np.isnan(selected_lai_flatten))[0]
    
        if len(positions) > 0:
            for position in positions:
                selected_lai_flatten[position] = (selected_lai_flatten[position-1]+selected_lai_flatten[position+1])/2 #filling with the average of two nearest value
            gap_free_lai = selected_lai_flatten.reshape(-1, 46)
            return gap_free_lai,selected_dates
    else:
        return selected_lai,selected_dates