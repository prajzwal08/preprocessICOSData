import numpy as np
import pandas as pd

def meter_to_milimeter(variable_in_meter:np.ndarray) -> np.ndarray:
    """
    Convert any variable in meter to mm.
    
    Args:
        variable_meter (np.ndarray): Array of any variable in meter.

    Returns:
        np.ndarray: Array of variable in mm.
    """
    variable_in_mm = variable_in_meter*1000
    return variable_in_mm
    
    

def kelvin_to_celsius(temperature_kelvin: np.ndarray) -> np.ndarray:
    """
    Convert temperatures from Kelvin to Celsius using NumPy arrays.

    Args:
        temperature_kelvin (np.ndarray): Array of temperatures in Kelvin.

    Returns:
        np.ndarray: Array of temperatures in Celsius.
    """
    if np.any(temperature_kelvin < 0):
        raise ValueError("Temperature in Kelvin cannot be negative.")
    
    temperature_celsius = temperature_kelvin - 273.15
    return temperature_celsius

def pascal_to_hectoPascal(variable_pascal: np.ndarray) -> np.ndarray:
    """
    Convert pressure values from pascal to hectopascal (hPa) using NumPy arrays.

    Args:
        variable_pascal (np.ndarray): Array of pressure values in pascal.

    Returns:
        np.ndarray: Array of pressure values in hectopascal (hPa).
    """
    
    variable_hectopascal = variable_pascal / 100
    return variable_hectopascal


def pascal_to_hectoPascal(variable_pascal: np.ndarray) -> np.ndarray:
    """
    Convert pressure values from pascal to hectopascal (hPa) using NumPy arrays.

    Args:
        variable_pascal (np.ndarray): Array of pressure values in pascal.

    Returns:
        np.ndarray: Array of pressure values in hectopascal (hPa).
    """
    
    variable_hectopascal = variable_pascal / 100
    return variable_hectopascal


def kilopascal_to_hectoPascal(variable_kilopascal: np.ndarray) -> np.ndarray:
    """
    Convert pressure values from kilopascal to hectopascal (hPa) using NumPy arrays.

    Args:
        variable_kilopascal (np.ndarray): Array of pressure values in kilopascal.

    Returns:
        np.ndarray: Array of pressure values in hectopascal (hPa).
    """
    
    # Convert kilopascal to hectopascal
    variable_hectopascal = variable_kilopascal * 10
    return variable_hectopascal


def joule_to_watt(variable_joule: np.ndarray, accumulation_period: float, unit: str = "s") -> np.ndarray:
    """
    Convert energy from joules to power in watts, considering different accumulation units.

    Args:
        variable_joule (np.ndarray): Array of energy values in joules.
        accumulation_period (float): The period over which the energy was accumulated.
        unit (str): The unit of the accumulation period ("h" for hours, "m" for minutes, "s" for seconds).
                    Default is "s" for seconds.

    Returns:
        np.ndarray: Array of power values in watts.
    """
    # Convert accumulation period to seconds based on the unit
    if unit == "h":
        period_seconds = accumulation_period * 3600  # hours to seconds
    elif unit == "m":
        period_seconds = accumulation_period * 60  # minutes to seconds
    elif unit == "s":
        period_seconds = accumulation_period  # already in seconds
    else:
        raise ValueError("Invalid unit. Please use 'h' for hours, 'm' for minutes, or 's' for seconds.")

    # Calculate the power in watts
    variable_watt = variable_joule / period_seconds
    return variable_watt

def convert_local_to_utc(local_times: np.ndarray, time_difference: int) -> np.ndarray:
    """
    Convert local times to UTC based on a fixed time difference without considering daylight saving.

    Parameters:
    - local_times (np.ndarray): An array of local datetime values as numpy datetime64.
    - time_difference (int): The time difference relative to UTC in hours (e.g., +1 for UTC+1, -5 for UTC-5).

    Returns:
    - np.ndarray: The datetime values converted to UTC.
    """
    # Convert the NumPy array to pandas DatetimeIndex for handling timedelta
    date_times = pd.DatetimeIndex(local_times)

    # Convert local time to UTC by subtracting the time difference
    utc_times = date_times - pd.Timedelta(hours=time_difference)

    # Convert back to NumPy array
    return pd.to_datetime(utc_times.to_numpy())

