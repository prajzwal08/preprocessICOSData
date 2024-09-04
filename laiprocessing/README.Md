## Folder Structure

The project directory is organized as follows:

- **comparison**: Contains scripts for comparing LAI data.
  - `lai_comparison.py`: Compares LAI data from different sources, mainly copernicusLAIv0,v3 and MODISLAI

- **download**: Scripts for downloading LAI products.
  - `download_lai_products.py`: Downloads LAI data products.

- **plot_lai_comparison**: Scripts for plotting LAI comparison results.
  - `plot_lai_comparison.py`: Generates comparison plots.

- **preprocesscopernicuslai.py**: Preprocesses Copernicus LAI data. 

- **preprocessmodisLAI.py**: Preprocesses MODIS LAI data.

- **utilspy**: Utility functions for preprocessing.
  - `utils.py`: Contains utility functions utilised in both preprocessing.

- **main.py**: Main script that ties together the entire workflow.

## Output
1).csv file with date and LAI 
2) plots making comparison
