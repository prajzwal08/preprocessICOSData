import cdsapi
import os

# Initialize CDS API client
c = cdsapi.Client()

# Generate list of years from 2003 to 2020
years = list(range(2003, 2021))

# Generate list of months from 1 to 12
months = list(range(1, 13))

# Output path to save downloaded files
output_path = "/home/khanalp/data/copernicus_lai"

# Function to get the days that the FAPAR LAI is available for a certain year and month
def get_lai_days(year, month):
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return ["10", "20", "31"]
    if month == 2:
        if year in [2000, 2004, 2008, 2012, 2016, 2020]:
            return ["10", "20", "29"]
        else:
            return ["10", "20", "28"]
    return ["10", "20", "30"]



# Loop through years and months to retrieve FAPAR LAI data
for year in years:
    for month in months:
        # Construct request for data retrieval
        request = {
            'format': 'zip',
            'variable': 'lai',
            'horizontal_resolution': '1km',
            'satellite': 'spot' if year < 2014 else 'proba',
            'sensor': 'vgt',
            'product_version': 'V3',
            'year': f'{year}',
            'month': f'{month:0>2}',
            'nominal_day': get_lai_days(year, month),
            'area': [
                71.18, -25, 35.81, 44.79,
            ],
        }
        
        # Retrieve data from CDS
        c.retrieve('satellite-lai-fapar', request, os.path.join(output_path, f"satellite-lai-fapar_europe_{year}-{month}.zip"))
        
