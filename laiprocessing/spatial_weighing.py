import numpy as np

def get_spatial_weighted_LAI(lai_pixel, sd_pixel, qc_pixel):
    """
    Calculates the spatially weighted Leaf Area Index (LAI) based on the input LAI values, standard deviation (SD) values, and quality control (QC) flags.
    
    Parameters:
        lai_pixel (numpy.ndarray): Array containing the LAI values for each pixel.
        sd_pixel (numpy.ndarray): Array containing the standard deviation values for each pixel.
        qc_pixel (numpy.ndarray): Array containing the quality control flags for each pixel.
    
    Returns:
        numpy.ndarray: Array containing the spatially weighted LAI values.
    """
    # Use only good quality data
    lai_copy = np.copy(lai_pixel)
    sd_copy = np.copy(sd_pixel)

    # poor data quality flags 
    qc_flags = [0, 2, 24, 26, 32, 34, 56, 58]
    
    # Mask the quality flag
    mask = np.isin(qc_pixel, qc_flags)
    lai_copy[~mask] = np.nan
    sd_copy[~mask] = np.nan
    
    # Mask out where sd is really low (likely cloud effects)
    sd_copy[sd_copy < 0.1] = np.nan
    lai_copy[np.isnan(sd_copy)] = np.nan
    
    # Set the values above threshold to missing
    lai_copy[lai_copy > 10] = np.nan
    
    # Calculate weights, ignoring NaN values
    weights = (1 / sd_copy**2) / np.nansum(1 / sd_copy**2, axis=1, keepdims=True)
    
    # Element-wise multiplication of lai_pixel and weights
    weighted_lai_values = lai_copy * weights
    
    # Calculate the weighted mean for each row, ignoring NaN values
    weighted_lai = np.nansum(weighted_lai_values, axis = 1)
    
    return weighted_lai