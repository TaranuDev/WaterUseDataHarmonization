import regionmask

import cartopy.crs as ccrs

import h5netcdf

import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import os

import warnings
warnings.filterwarnings('ignore')

def plot_countries_monthly_timeseries_no_correction(path_saving_plots, hist_countries_timeseries, future_countries_timeseries, sector_name, countries_names):
    
    # Generate time ranges for the historical and future datasets in monthly intervals
    times_historical = pd.date_range(start='1971-01', periods=hist_countries_timeseries.sizes['time'], freq='M')
    times_future = pd.date_range(start='2011-01', periods=future_countries_timeseries[:, 12:].sizes['time'], freq='M') # skip first year as it is matching with the 2010 in the historical


    # Iterate over each country by its name
    for country_name in hist_countries_timeseries.country.values:
        # Extracting the time series for the current country
        historical = hist_countries_timeseries.sel(country=country_name)
        future = future_countries_timeseries[:, 12:].sel(country=country_name) # skip first year as it is matching with the 2010 in the historical

        # Creating the plot
        plt.figure(figsize=(10, 6))
        plt.plot(times_historical, historical*((30.44*24*3600)/10**9), label='Historical', color='blue') # Transform units from m3/s to km3/month
        plt.plot(times_future, future*((30.44*24*3600)/10**9), label='Future', color='red') # Transform units from m3/s to km3/month

        plt.title(f"Water Withdrawal for " + sector_name + f" Sector - {countries_names[country_name]}")
        plt.xlabel("Date")
        plt.ylabel("Water Withdrawal (km³/month)")
        plt.legend()

        # Save the plot as a PDF with the country name in the filename
        plt.savefig(os.path.join(path_saving_plots, f"{countries_names[country_name]}_{sector_name}_water_withdrawal.pdf"), format='pdf')
        plt.close()  # Close the plot to free memory
    
    print("All plots have been created. You can check the results in the " + path_saving_plots + " directory.")
    
def plot_countries_monthly_timeseries_with_future_correction(path_saving_plots, hist_countries_timeseries, future_countries_timeseries, corrected_future_countries_timeseries, sector_name, countries_names):
    
    # Generate time ranges for the historical and future datasets in monthly intervals
    times_historical = pd.date_range(start='1971-01', periods=hist_countries_timeseries.sizes['time'], freq='M')
    times_future = pd.date_range(start='2011-01', periods=future_countries_timeseries[:, 12:].sizes['time'], freq='M') # skip first year as it is matching with the 2010 in the historical


    # Iterate over each country by its name
    for country_name in hist_countries_timeseries.country.values:
        # Extracting the time series for the current country
        historical = hist_countries_timeseries.sel(country=country_name)
        future = future_countries_timeseries[:, 12:].sel(country=country_name) # skip first year as it is matching with the 2010 in the historical
        future_corrected = corrected_future_countries_timeseries[:, 12:].sel(country=country_name) # skip first year as it is matching with the 2010 in the historical
        
        # Creating the plot
        plt.figure(figsize=(10, 6))
        plt.plot(times_historical, historical*((30.44*24*3600)/10**9), label='Historical', color='blue') # Transform units from m3/s to km3/month
        plt.plot(times_future, future*((30.44*24*3600)/10**9), label='Future', color='red') # Transform units from m3/s to km3/month
        plt.plot(times_future, future_corrected*((30.44*24*3600)/10**9), label='Future Corrected', color='green')
        
        plt.title(f"Water Withdrawal correction for " + sector_name + f" Sector - {countries_names[country_name]}")
        plt.xlabel("Date")
        plt.ylabel("Water Withdrawal (km³/month)")
        plt.legend()

        # Save the plot as a PDF with the country name in the filename
        plt.savefig(os.path.join(path_saving_plots, f"{countries_names[country_name]}_{sector_name}_water_withdrawal_corrected.pdf"), format='pdf')
        plt.close()  # Close the plot to free memory
    
    print("All plots have been created. You can check the results in the " + path_saving_plots + " directory.")
    
def plot_US_states_monthly_timeseries_no_correction(path_saving_plots, hist_US_states_timeseries, future_US_states_timeseries, sector_name, states_names):
    
    # Generate time ranges for the historical and future datasets in monthly intervals
    times_historical = pd.date_range(start='1971-01', periods=hist_US_states_timeseries.sizes['time'], freq='M')
    times_future = pd.date_range(start='2011-01', periods=future_US_states_timeseries[:, 12:].sizes['time'], freq='M') # skip first year as it is matching with the 2010 in the historical


    # Iterate over each state in US by its name
    for state_name in hist_US_states_timeseries.US_states.values:
        # Extracting the time series for the current state
        historical = hist_US_states_timeseries.sel(US_states=state_name)
        future = future_US_states_timeseries[:, 12:].sel(US_states=state_name) # skip first year as it is matching with the 2010 in the historical

        # Creating the plot
        plt.figure(figsize=(10, 6))
        plt.plot(times_historical, historical*((30.44*24*3600)/10**9), label='Historical', color='blue') # Transform units from m3/s to km3/month
        plt.plot(times_future, future*((30.44*24*3600)/10**9), label='Future', color='red') # Transform units from m3/s to km3/month

        plt.title(f"Water Withdrawal for " + sector_name + f" Sector - {states_names[state_name]}")
        plt.xlabel("Date")
        plt.ylabel("Water Withdrawal (km³/month)")
        plt.legend()

        # Save the plot as a PDF with the country name in the filename
        plt.savefig(os.path.join(path_saving_plots, f"US_{states_names[state_name]}_{sector_name}_water_withdrawal.pdf"), format='pdf')
        plt.close()  # Close the plot to free memory

    print("All plots have been created. You can check the results in the " + path_saving_plots + " directory.")
    
def plot_US_states_monthly_timeseries_with_future_correction(path_saving_plots, hist_US_states_timeseries, future_US_states_timeseries, corrected_future_US_states_timeseries, sector_name, states_names):
    
    # Generate time ranges for the historical and future datasets in monthly intervals
    times_historical = pd.date_range(start='1971-01', periods=hist_US_states_timeseries.sizes['time'], freq='M')
    times_future = pd.date_range(start='2011-01', periods=future_US_states_timeseries[:, 12:].sizes['time'], freq='M') # skip first year as it is matching with the 2010 in the historical


    # Iterate over each state in US by its name
    for state_name in hist_US_states_timeseries.US_states.values:
        # Extracting the time series for the current state
        historical = hist_US_states_timeseries.sel(US_states=state_name)
        future = future_US_states_timeseries[:, 12:].sel(US_states=state_name) # skip first year as it is matching with the 2010 in the historical
        future_corrected = corrected_future_US_states_timeseries[:, 12:].sel(US_states=state_name) # skip first year as it is matching with the 2010 in the historical
        
        # Creating the plot
        plt.figure(figsize=(10, 6))
        plt.plot(times_historical, historical*((30.44*24*3600)/10**9), label='Historical', color='blue') # Transform units from m3/s to km3/month
        plt.plot(times_future, future*((30.44*24*3600)/10**9), label='Future', color='red') # Transform units from m3/s to km3/month
        plt.plot(times_future, future_corrected*((30.44*24*3600)/10**9), label='Future Corrected', color='green')

        plt.title(f"Water Withdrawal correction for " + sector_name + f" Sector - {states_names[state_name]}")
        plt.xlabel("Date")
        plt.ylabel("Water Withdrawal (km³/month)")
        plt.legend()

        # Save the plot as a PDF with the country name in the filename
        plt.savefig(os.path.join(path_saving_plots, f"US_{states_names[state_name]}_{sector_name}_water_withdrawal_corrected.pdf"), format='pdf')
        plt.close()  # Close the plot to free memory

    print("All plots have been created. You can check the results in the " + path_saving_plots + " directory.")
    
def plot_global_monthly_timeseries_with_future_correction(path_saving_plots, hist_countries_timeseries, future_countries_timeseries, corrected_future_countries_timeseries, hist_US_states_timeseries, future_US_states_timeseries, corrected_future_US_states_timeseries, sector_name, countries_names, states_names):
    # Generate time ranges for the historical and future datasets in monthly intervals
    times_historical = pd.date_range(start='1971-01', periods=hist_countries_timeseries.sizes['time'], freq='M')
    times_future = pd.date_range(start='2011-01', periods=future_countries_timeseries[:, 12:].sizes['time'], freq='M') # skip first year as it is matching with the 2010 in the historical
    
    # Remove non US data from the states dataset
    states_no_world = hist_US_states_timeseries.sel(US_states=[c for c in hist_US_states_timeseries.US_states.values if c != 51])
    future_states_no_world = future_US_states_timeseries[:, 12:].sel(US_states=[c for c in future_US_states_timeseries.US_states.values if c != 51])
    corrected_future_states_no_world = corrected_future_US_states_timeseries[:, 12:].sel(US_states=[c for c in corrected_future_US_states_timeseries.US_states.values if c != 51])

    # Aggregate US states data to a single time series
    hist_US = states_no_world.sum(dim='US_states')
    future_US = future_states_no_world.sum(dim='US_states') 
    corrected_future_US = corrected_future_states_no_world.sum(dim='US_states') 
    
    # Remove the US data from the countries dataset
    countries_no_us = hist_countries_timeseries.sel(country=[c for c in hist_countries_timeseries.country.values if c != 4])
    future_countries_no_us = future_countries_timeseries[:, 12:].sel(country=[c for c in future_countries_timeseries.country.values if c != 4])
    corrected_future_countries_no_us = corrected_future_countries_timeseries[:, 12:].sel(country=[c for c in corrected_future_countries_timeseries.country.values if c != 4])
    
    # Sum over the remaining countries
    hist_countries_global = countries_no_us.sum(dim='country')
    future_countries_global = future_countries_no_us.sum(dim='country')
    corrected_future_countries_global = corrected_future_countries_no_us.sum(dim='country')
    
    # Combine US and the rest of the world
    hist_global = hist_countries_global + hist_US
    future_global = future_countries_global + future_US
    corrected_future_global = corrected_future_countries_global + corrected_future_US
    
    # Plot the global data
    plt.figure(figsize=(12, 8))
    plt.plot(times_historical, hist_global*((30.44*24*3600)/10**9), label='Historical', color='blue') # Transform units from m3/s to km3/month
    # plt.plot(times_future, future_global*((30.44*24*3600)/10**9), label='Future', color='red') # Transform units from m3/s to km3/month
    plt.plot(times_future, corrected_future_global*((30.44*24*3600)/10**9), label='Future Corrected', color='green')
    
    plt.title(f"Global Water Withdrawal correction for {sector_name} Sector")
    plt.xlabel("Date")
    plt.ylabel("Water Withdrawal (km³/month)")
    plt.legend()
    
    # Save the plot
    plt.savefig(os.path.join(path_saving_plots, f"Global_{sector_name}_water_withdrawal_corrected.pdf"), format='pdf')
    plt.close()
    
    print(f"Global plot has been created. You can check the results in the {path_saving_plots} directory.")

    
def plot_global_monthly_timeseries_no_correction(path_saving_plots, hist_countries_timeseries, future_countries_timeseries, hist_US_states_timeseries, future_US_states_timeseries, sector_name, countries_names, states_names):
    # Generate time ranges for the historical and future datasets in monthly intervals
    times_historical = pd.date_range(start='1971-01', periods=hist_countries_timeseries.sizes['time'], freq='M')
    times_future = pd.date_range(start='2011-01', periods=future_countries_timeseries[:, 12:].sizes['time'], freq='M') # skip first year as it is matching with the 2010 in the historical
    
    # Remove non US data from the states dataset
    states_no_world = hist_US_states_timeseries.sel(US_states=[c for c in hist_US_states_timeseries.US_states.values if c != 51])
    future_states_no_world = future_US_states_timeseries[:, 12:].sel(US_states=[c for c in future_US_states_timeseries.US_states.values if c != 51])

    # Aggregate US states data to a single time series
    hist_US = states_no_world.sum(dim='US_states')
    future_US = future_states_no_world.sum(dim='US_states') 
    
    # Remove the US data from the countries dataset
    countries_no_us = hist_countries_timeseries.sel(country=[c for c in hist_countries_timeseries.country.values if c != 4])
    future_countries_no_us = future_countries_timeseries[:, 12:].sel(country=[c for c in future_countries_timeseries.country.values if c != 4])
    
    # Sum over the remaining countries
    hist_countries_global = countries_no_us.sum(dim='country')
    future_countries_global = future_countries_no_us.sum(dim='country')
    
    # Combine US and the rest of the world
    hist_global = hist_countries_global + hist_US
    future_global = future_countries_global + future_US
    
    # Plot the global data
    plt.figure(figsize=(12, 8))
    plt.plot(times_historical, hist_global*((30.44*24*3600)/10**9), label='Historical', color='blue') # Transform units from m3/s to km3/month
    plt.plot(times_future, future_global*((30.44*24*3600)/10**9), label='Future', color='red') # Transform units from m3/s to km3/month
    
    plt.title(f"Global Water Withdrawal for {sector_name} Sector")
    plt.xlabel("Date")
    plt.ylabel("Water Withdrawal (km³/month)")
    plt.legend()
    
    # Save the plot
    plt.savefig(os.path.join(path_saving_plots, f"Global_{sector_name}_water_withdrawal.pdf"), format='pdf')
    plt.close()
    
    print(f"Global plot has been created. You can check the results in the {path_saving_plots} directory.")

    
def plot_water_withdrawal_changes(historical_data, corrected_or_not_future_data, variable_name_hist, variable_name_future, variable_name_for_the_title, path_save_plot):
    map_proj = ccrs.Robinson(central_longitude=0)
    years = range(2007, 2013)
    
    fig, axes = plt.subplots(2, 3, figsize=(30, 20), subplot_kw={'projection': map_proj})
    axes = axes.flatten()

    for i, year in enumerate(years):
        ax = axes[i]

        if year < 2010:
            ww_year = historical_data.sel(time=str(year))[variable_name_hist][0]
            ww_year2 = historical_data.sel(time=str(year + 1))[variable_name_hist][0]
        elif year == 2010:
            ww_year = historical_data.sel(time=str(year))[variable_name_hist][0]
            ww_year2 = corrected_or_not_future_data.sel(time=str(year + 1))[variable_name_future][0]
        else:
            ww_year = corrected_or_not_future_data.sel(time=str(year))[variable_name_future][0]
            ww_year2 = corrected_or_not_future_data.sel(time=str(year + 1))[variable_name_future][0]

        delta = (ww_year2 - ww_year) / ww_year * 100

        ax.axis('off')
        ax.coastlines(color='lightgray', linewidth=0.5)
        delta.plot(ax=ax, cbar_kwargs={'fraction': 0.02, 'pad': 0.04}, cmap='RdBu', vmin=-30, vmax=30, transform=ccrs.PlateCarree())
        
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title('')
        ax.set_title(f'$\Delta$ {variable_name_for_the_title} {year}-{year + 1} (%)', loc='right', fontsize=20)

    fig.tight_layout()
    plt.savefig(path_save_plot, format='pdf')
    
def plot_spatial_consistency_for_countries(countries_names, spatial_consistency_original, spatial_consistency_corrected,  save_directory, variable_name):
    years = pd.date_range(start=str(spatial_consistency_corrected.time.values[0]), periods=len(spatial_consistency_corrected.time.values), freq='Y').year
    
    # Small offset to ensure visibility of the lines when the value is 1
    offset = 0.01
    
    # Iterate over each country and its population time series
    for i, country in enumerate(countries_names):
        original_data = spatial_consistency_original[i]
        corrected_data = spatial_consistency_corrected[i]

        # Apply offset to the data if they are exactly 1
        original_data = [val + offset if val == 1 else val for val in original_data]
        corrected_data = [val + offset if val == 1 else val for val in corrected_data]

        # Creating a DataFrame for easy manipulation
        df = pd.DataFrame({
            'Year': years, 
            'Spatial Consistency Original': original_data,
            'Spatial Consistency Corrected': corrected_data
        })
        

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(df['Year'], df['Spatial Consistency Original'], 'b-', label='Original', markersize=3)
        plt.plot(df['Year'], df['Spatial Consistency Corrected'], 'r-', label='Corrected National Timeseries', markersize=3)
        plt.axvline(x=2010, color='green', linestyle='--')  # This line adds a vertical line at the year 2010
        plt.title(country)
        plt.xlabel('Year')
        plt.ylabel('Spatial Consistency Metric')
        plt.ylim(bottom=1)  # Set the minimum value of the y-axis to 1
        plt.legend()

        # Save to pdf
        pdf_path = f"{save_directory}/{variable_name}_{country}_spatial_consistency_metric_1971_2100.pdf"
        plt.savefig(pdf_path, format='pdf')
        plt.close()

        
def plot_spatial_consistency_for_US_states(states_names, spatial_consistency_original, spatial_consistency_corrected,  save_directory):
    years = pd.date_range(start=str(spatial_consistency_corrected.time.values[0]), periods=len(spatial_consistency_corrected.time.values), freq='Y').year

    # Iterate over each country and its population time series
    for i, state in enumerate(states_names):
        original_data = spatial_consistency_original[i]
        corrected_data = spatial_consistency_corrected[i]

        # Creating a DataFrame for easy manipulation
        df = pd.DataFrame({
            'Year': years, 
            'Spatial Consistency Original': original_data,
            'Spatial Consistency Corrected': corrected_data
        })
        

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(df['Year'], df['Spatial Consistency Original'], 'b-', label='Original', markersize=3)
        plt.plot(df['Year'], df['Spatial Consistency Corrected'], 'r-', label='Corrected State Timeseries', markersize=3)
        plt.axvline(x=2010, color='green', linestyle='--')  # This line adds a vertical line at the year 2010
        plt.title(state)
        plt.xlabel('Year')
        plt.ylabel('Spatial Consistency Metric')
        plt.ylim(bottom=1)  # Set the minimum value of the y-axis to 1
        plt.legend()

        # Save to pdf
        pdf_path = f"{save_directory}/{state}_spatial_consistency_metric_1971_2100.pdf"
        plt.savefig(pdf_path, format='pdf')
        plt.close()


        
        
def plot_water_withdrawal_changes(historical_data, original_future_data, corrected_future_data, variable_name_hist, variable_name_future, variable_name_for_the_title, path_save_plot):
    map_proj = ccrs.Robinson(central_longitude=0)
    
    # Create a figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': map_proj})
    
    # First figure: Change for 2010/2011 with the original data
    ax1 = axes[0]
    ww_2010 = historical_data.sel(time='2010')[variable_name_hist][0]
    ww_2011_original = original_future_data.sel(time='2011')[variable_name_future][0]
    delta_original = (ww_2011_original - ww_2010) / ww_2010 * 100
    
    ax1.axis('off')
    ax1.coastlines(color='lightgray', linewidth=0.5)
    delta_original.plot(ax=ax1, cbar_kwargs={'fraction': 0.02, 'pad': 0.04}, cmap='RdBu', vmin=-30, vmax=30, transform=ccrs.PlateCarree(), add_colorbar=True, add_labels=False)
    ax1.set_title(f'$\Delta$ {variable_name_for_the_title} 2010-2011 (Original) (%)', loc='right', fontsize=20)
    
    # Second figure: Change for 2010/2011 with the corrected data
    ax2 = axes[1]
    ww_2011_corrected = corrected_future_data.sel(time='2011')[variable_name_future][0]
    delta_corrected = (ww_2011_corrected - ww_2010) / ww_2010 * 100
    
    ax2.axis('off')
    ax2.coastlines(color='lightgray', linewidth=0.5)
    delta_corrected.plot(ax=ax2, cbar_kwargs={'fraction': 0.02, 'pad': 0.04}, cmap='RdBu', vmin=-30, vmax=30, transform=ccrs.PlateCarree(), add_colorbar=True, add_labels=False)
    ax2.set_title(f'$\Delta$ {variable_name_for_the_title} 2010-2011 (Corrected) (%)', loc='right', fontsize=20)
    
    fig.tight_layout()
    plt.savefig(path_save_plot, format='pdf')
