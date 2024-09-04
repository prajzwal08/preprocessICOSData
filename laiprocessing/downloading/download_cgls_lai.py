from terracatalogueclient import Catalogue
from terracatalogueclient.config import CatalogueConfig, CatalogueEnvironment
import pandas as pd
import datetime as dt

#From https://github.com/cgls/globalland_API_demo_python/blob/main/search_and_download/CGLS_catalogue_and_download_demo.ipynb

# Specify the output folder where the downloaded products will be saved
output_folder = "/home/khanalp/data/copernicus_lai/cgls"

# Initialize the configuration for accessing the catalogue
config = CatalogueConfig.from_environment(CatalogueEnvironment.CGLS)
catalogue = Catalogue(config)

# List to store information about available products
rows = []

# Get information about products available within the specified time range
products = catalogue.get_products(
    "clms_global_lai_1km_v2_10daily_netcdf",  # Product identifier
    start=dt.date(2003, 1, 1),  # Start date
    end=dt.date(2019, 12, 31)    # End date
)

# Iterate over each product to gather its details
for product in products:
    # Append product details to the rows list
    rows.append([product.id, product.data[0].href, (product.data[0].length/(1024*1024))])

# Create a DataFrame from the collected product details
df = pd.DataFrame(data=rows, columns=['Identifier', 'URL', 'Size (MB)'])

# Retrieve a list of products to download
product_list = list(catalogue.get_products(
    "clms_global_lai_1km_v2_10daily_netcdf",  # Product identifier
    start=dt.date(2003, 1, 1),  # Start date
    end=dt.date(2019, 12, 31)    # End date
))

# Download the products to the specified output folder
catalogue.download_products(product_list, output_folder, force= True)

# Print a message indicating that all downloads are completed
print("All downloads completed.")

