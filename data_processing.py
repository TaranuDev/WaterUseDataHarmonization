import os
import xarray as xr
import numpy as np
import pandas as pd
import h5netcdf
import regionmask
import warnings
warnings.filterwarnings('ignore')

def process_files(directory, pattern):
    all_years = []
    
    # Loop through each file in the directory that matches the pattern
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".nc") and pattern in filename:
            full_path = os.path.join(directory, filename)
            # Read the .nc file
            ds = xr.open_dataset(full_path)
            
            # Append the new data to the list
            all_years.append(ds)
    
    # Combine the list content into a single dataset
    combined = xr.concat(all_years, dim='time')
    
    print("A xarray.Dataset corresponding to the data in the given path was succesfully generated.")
    return combined

def process_files_annual_average(directory, pattern):
    all_years = []
    
    # Loop through each file in the directory that matches the pattern
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".nc") and pattern in filename:
            full_path = os.path.join(directory, filename)
            # Read the .nc file
            ds = xr.open_dataset(full_path)
            
            # Calculate the yearly average
            yearly_avg = ds.resample(time='A').mean()
            
            # Append the yearly average to the list
            all_years.append(yearly_avg)
    
    # Combine all yearly averages into a single dataset
    combined = xr.concat(all_years, dim='time')
    
    print("A xarray.Dataset corresponding to the annualy aggregated data in the given path was succesfully generated.")
    return combined


def read_mask_file(path):
    mask = xr.open_dataset(path).mask.values
    print("A numpy.ndarray corresponding to the mask file provided was succesfully generated.")
    return mask


def define_countries_and_states_names_and_indices():
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
    states = regionmask.defined_regions.natural_earth_v5_0_0.us_states_50
    
    # Obtain a list of all countries names and indices:
    countries_names = countries.names
    countries_indices = countries.numbers

    # Obtain a list of all US states names and indices:
    states_names = states.names
    states_indices = states.numbers
    
    # Create a fictive country which would contain all the 177 masked gridcells
    countries_names.append("Not a regionmask country")
    countries_indices.append(177)
    
    # Create a fictive US state which would contain all the 51 masked gridcells
    states_names.append("Not a regionmask state")
    states_indices.append(51)
    
    print("Countries and US states names were initialized. Country index 177 corresponds to gridcells not attributed to any country. State code 51 corresponds to gridcells not attributed to any US state.")
    return countries_names, countries_indices, states_names, states_indices


def compute_national_or_state_monthly_timeseries(input_gridded_data, output_timeseries_dataArray, mask_of_countries_or_states):
    
    # Get the first and last year we need to itterate over:
    time_data = output_timeseries_dataArray['time']  # Accessing the time coordinate
    time_data_pd = pd.to_datetime(time_data.values)
    
    first_year = time_data_pd[0].year
    last_year = time_data_pd[-1].year
    
    print("Provided data covers the years " + str(first_year) + "--" + str(last_year) + ".")
    if np.nanmax(mask_of_countries_or_states) < 52 and last_year < 2011:
        print("Begin computing the historical US states time-series.")
    if np.nanmax(mask_of_countries_or_states) < 52 and last_year > 2011:
        print("Begin computing the future US states time-series.")
    if np.nanmax(mask_of_countries_or_states) > 52 and last_year < 2011:
        print("Begin computing the historical countries time-series.")
    if np.nanmax(mask_of_countries_or_states) > 52 and last_year > 2011:
        print("Begin computing the future countries time-series.")
        
    # Calculate total iterations
    total_iterations = last_year - first_year + 1

    # Calculate the step size for each 25% increment
    step_size = total_iterations // 4  # Use integer division to ensure whole number steps

    
    # Get the variable name contained in the input_gridded_data
    variable_name = list(input_gridded_data.data_vars)[0]  # This will get the first (and only) variable's name

    # Loop over each year:
    # for year in range(first_year, last_year+1):
    for i, year in enumerate(range(first_year, last_year + 1), start=1):
        # The data is in monthly format, so loop over each month of the year
        for month in range(1, 13):  # 12 months
            month_idx = (year - first_year) * 12 + (month - 1)  # Calculate the index for the month in the array
            
            # Loop over each country or state (depending on the provided output_timeseries_dataArray)
            for country_or_state in range(len(output_timeseries_dataArray)):
                # Calculate country/state sum for given year and month
                output_timeseries_dataArray[country_or_state, month_idx] = np.nansum(input_gridded_data[variable_name][month_idx].where(mask_of_countries_or_states == country_or_state))
        
        # Print a progress update
        if i % step_size == 0 or i == total_iterations:
            percent_complete = (i / total_iterations) * 100
            print(f"Calculation completed at {int(percent_complete)}%")
            

            
# Function to check for negative values in a dataset
def check_negative_values(dataarray, names, entity_type="entity"):
    negatives_found = False
    for index in range(len(names)):
        # Select the data for the current country or state
        data = dataarray.sel({entity_type: index})
        
        # Check if any negative values exist
        if (data < 0).any():
            negatives_found = True
            print(f"Negative values found in {names[index]}")
    
    if not negatives_found:
        print(f"No negative values found in {entity_type.capitalize()} dataset.")