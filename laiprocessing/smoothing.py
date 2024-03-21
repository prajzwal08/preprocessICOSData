import numpy as np
from scipy.signal import savgol_filter

# Function to calculate rolling mean along rows
def rolling_mean(array, window):
    """
    Calculates the rolling mean along a 1D numpy array.
    
    Parameters:
        array (numpy.ndarray): Input 1D array.
        window (int): Size of the rolling window.
    
    Returns:
        numpy.ndarray: Resultant array after applying the rolling mean.
    """
    result = np.full_like(array, np.nan)
    for i in range(array.shape[0]):
        start_index = max(0, i - window // 2)
        end_index = min(array.shape[0], i + (window + 1) // 2)
        result[i] = np.mean(array[start_index:end_index], axis=0)
    return result

def smoothing_LAI(gap_free_lai):
    """
    Smoothes the gap-free LAI data by calculating climatology, removing mean climatology to obtain anomalies,
    and applying a rolling mean to the anomalies with a window of +/- 6 months. Finally, it adds the smoothed anomalies
    back to the climatology to obtain smoothed LAI values.
    
    Parameters:
        gap_free_lai (numpy.ndarray): Array containing gap-free LAI data.
    
    Returns:
        numpy.ndarray: Smoothed LAI values.
    """
    # Calculate the mean of each column ignoring NaNs (climatology)
    column_means = np.nanmean(gap_free_lai, axis=0)
    #smoothed_column_means = savgol_filter(column_means, window_length = 13, polyorder = 3)
    # Calculate the anomaly after removing mean climatology. 
    anomaly = gap_free_lai - column_means
    # Calculate running mean anomaly (+/- 6 months either side of each time step)
    anomaly_rolling = rolling_mean(anomaly.flatten(), 13)
    smoothed_lai = anomaly_rolling + np.tile(column_means,(anomaly.shape[0]))
    smoothed_lai_smoothed = savgol_filter(smoothed_lai, window_length = 13, polyorder = 3)
    smoothed_lai_smoothed[smoothed_lai_smoothed < 0] = 0
    return smoothed_lai_smoothed