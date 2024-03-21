import numpy as np
from scipy import interpolate

def interpolate_NA_LAI(unfilled_lai):
    """
    Interpolates missing values (NaNs) in a given array of LAI (Leaf Area Index) using cubic interpolation and caps negative values to zero.
    
    Parameters:
        unfilled_lai (numpy.ndarray): Array containing LAI values with missing values represented as NaNs.
    
    Returns:
        numpy.ndarray: Array with missing values filled using interpolation and negative values capped at zero.
    """
    filled_lai = unfilled_lai.copy()
    
    # Create a mask for NaN values
    nan_mask = np.isnan(unfilled_lai)
    
    # Generate an index array for values
    x = np.arange(len(unfilled_lai))
    
    # Interpolate only at the positions where NaNs are present
    interp_func = interpolate.interp1d(x[~nan_mask], unfilled_lai[~nan_mask], kind='cubic', fill_value="extrapolate")
    
    # Extrapolate the NaN values
    filled_lai[nan_mask] = interp_func(x)[nan_mask]
    
    # Cap negative values to zero
    filled_lai[filled_lai < 0] = 0
    
    # Set the last observation to zero
    filled_lai[-1] = 0
    
    return filled_lai