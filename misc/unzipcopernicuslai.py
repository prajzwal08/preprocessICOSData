"""
Author: Prajwal Khanal
Date: 09 April 2024 @ Hengelo Public library 
Purpose: Unzip files and organize them into folders based on the year in the file names
"""

import os
import zipfile

# Define the file path
file_path_copernicus_lai = "/home/khanalp/downloads/copernicuslai/"

# List all files in the directory
files = os.listdir(file_path_copernicus_lai)

# Filter out only the .zip files
zip_files = [file for file in files if file.endswith('.zip')]

# Ensure at least one .zip file is found
if zip_files:
    # Create folders for each year
    for year in range(2003, 2021):  # years from 2003 to 2020
        year_folder = os.path.join(file_path_copernicus_lai, str(year))
        os.makedirs(year_folder, exist_ok=True)

    # Iterate through each .zip file
    for zip_file in zip_files:
        # Get the path of the .zip file
        zip_file_path = os.path.join(file_path_copernicus_lai, zip_file)
        
        # Extract the year from the file name
        year = zip_file.split('_')[-1].split('.')[0].split('-')[0]
        
        # Destination folder based on the year
        year_folder = os.path.join(file_path_copernicus_lai, year)
        
        # Extract the contents of the .zip file into the year folder
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(year_folder)
        
        print(f"Contents of {zip_file} have been extracted to folder {year}.")
    
else:
    print("No .zip files found in the directory.")
