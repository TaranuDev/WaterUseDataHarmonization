import os
import xarray as xr
import numpy as np
import pandas as pd
import h5netcdf
import regionmask
import warnings
warnings.filterwarnings('ignore')


def min_max_rescale(data, new_min, new_max):
    """
    Perform min-max normalization on a list of numbers to a new specified range.
    """
    
    old_min = min(data).values
    old_max = max(data).values
    
    if new_min < 0:
        new_min = 0.0
    
    if old_min == old_max:
        return [new_min for _ in data]
    
    normalized_data = [
        new_min + (float(x - old_min) / (old_max - old_min)) * (new_max - new_min)
        for x in data
    ]
    
    return np.array(normalized_data)


import xarray as xr

def future_timeseries_correction_algorithm(historical_data, future_data_original, years_future,
                                           use_bias_adjustment=True, apply_amplitude_correction=True, 
                                           number_of_years_to_use_for_amplitude_reference_N=1, 
                                           apply_iterative_rescaling_to_correct_for_negative_values=True, 
                                           minimum_future_ratio=1.0):
    
    # Calculate the slice range for the past N years from the last available year in historical data
    last_year = historical_data.time.dt.year.max().values
    start_year = last_year - number_of_years_to_use_for_amplitude_reference_N + 1
    historical_period = historical_data.sel(time=slice(f'{start_year}-01', f'{last_year}-12'))
    
    # Calculate middle points and amplitudes for each year in the historical period
    middle_points = []
    amplitudes = []
    for year in range(start_year, last_year + 1):
        yearly_data = historical_period.sel(time=slice(f'{year}-01', f'{year}-12'))
        yearly_max = yearly_data.max(dim='time')
        yearly_min = yearly_data.min(dim='time')
        yearly_middle_point = (yearly_max + yearly_min) / 2
        yearly_amplitude = yearly_max - yearly_min
        middle_points.append(yearly_middle_point)
        amplitudes.append(yearly_amplitude)
    
    # Calculate the average middle point and amplitude over the past N years
    middle_point_historical = sum(middle_points) / len(middle_points)
    amplitude_historical_avg = sum(amplitudes) / len(amplitudes)
    
    # Calculate the mean middle point and amplitude for the year 2010 in future data
    future_2010 = future_data_original.sel(time=slice('2010-01', '2010-12'))
    middle_point_future = (future_2010.max() + future_2010.min()) / 2
    
    # Initialize the corrected future data array
    future_data_corrected = future_data_original.copy()
    
    # Adjustment based on the difference in middle points (Bias Adjustment)
    if use_bias_adjustment:
        adjustment = middle_point_historical - middle_point_future
        future_data_corrected = future_data_corrected + adjustment
    
    # Initialize the amplitude corrected data array
    future_data_corrected_including_amplitude_effect_xarray = xr.DataArray(data=future_data_corrected * 0.0, dims=["time"], coords={"time": years_future})

    if apply_amplitude_correction:
        for year in range(2010, future_data_corrected.time.dt.year.max().values + 1):
            future_year = future_data_corrected.sel(time=slice(f'{year}-01', f'{year}-12'))

            Min_n_corrected_with_adjustment_only = future_year.min().values
            Max_n_corrected_with_adjustment_only = future_year.max().values
            Middle_point_n_corrected_with_adjustment_only = (Max_n_corrected_with_adjustment_only + Min_n_corrected_with_adjustment_only) / 2.0
            A_n_corrected_with_adjustment_only = Max_n_corrected_with_adjustment_only - Min_n_corrected_with_adjustment_only

            if year == 2010:
                corrected_values = min_max_rescale(future_year, middle_point_historical - (amplitude_historical_avg / 2.0), middle_point_historical + (amplitude_historical_avg / 2.0))
                future_data_corrected_including_amplitude_effect_xarray.loc[dict(time=future_year.time)] = corrected_values
                A_n_previous_corrected_with_adjustment_and_rescaling = amplitude_historical_avg
                A_n_previous_corrected_with_adjustment_only = A_n_corrected_with_adjustment_only
            else:
                if A_n_corrected_with_adjustment_only == 0:
                    Min_n = Middle_point_n_corrected_with_adjustment_only
                    Max_n = Middle_point_n_corrected_with_adjustment_only
                else:
                    Min_n = Middle_point_n_corrected_with_adjustment_only - (A_n_previous_corrected_with_adjustment_and_rescaling / 2) * (A_n_corrected_with_adjustment_only / A_n_previous_corrected_with_adjustment_only)
                    Max_n = Middle_point_n_corrected_with_adjustment_only + (A_n_previous_corrected_with_adjustment_and_rescaling / 2) * (A_n_corrected_with_adjustment_only / A_n_previous_corrected_with_adjustment_only)
                corrected_values = min_max_rescale(future_year, Min_n, Max_n)
                future_data_corrected_including_amplitude_effect_xarray.loc[dict(time=future_year.time)] = corrected_values
                A_n_previous_corrected_with_adjustment_and_rescaling = Max_n - Min_n
                A_n_previous_corrected_with_adjustment_only = A_n_corrected_with_adjustment_only

    future_data_final = future_data_corrected_including_amplitude_effect_xarray if apply_amplitude_correction else future_data_corrected
    
    # Iterative rescaling to correct for negative values
    if apply_iterative_rescaling_to_correct_for_negative_values:
        future_data_final = iterative_rescaling(future_data_final, historical_data, middle_point_historical, minimum_future_ratio, start_year, last_year)
    
    print("Value from the correction algo:")
    print(np.nansum(future_data_final))

    return future_data_final

def future_timeseries_correction_algorithm_v2(historical_data, future_data_original, years_future,
                                           use_bias_adjustment=True, apply_amplitude_correction=True, 
                                           number_of_years_to_use_for_amplitude_reference_N=1, 
                                           apply_iterative_rescaling_to_correct_for_negative_values=True, 
                                           minimum_future_ratio=1.0):
    
    # Calculate the slice range for the past N years from the last available year in historical data
    last_year = historical_data.time.dt.year.max().values
    start_year = last_year - number_of_years_to_use_for_amplitude_reference_N + 1
    historical_period = historical_data.sel(time=slice(f'{start_year}-01', f'{last_year}-12'))
    
    # Calculate middle points and amplitudes for each year in the historical period
    middle_points = []
    amplitudes = []
    for year in range(start_year, last_year + 1):
        yearly_data = historical_period.sel(time=slice(f'{year}-01', f'{year}-12'))
        yearly_max = yearly_data.max(dim='time')
        yearly_min = yearly_data.min(dim='time')
        yearly_middle_point = (yearly_max + yearly_min) / 2
        yearly_amplitude = yearly_max - yearly_min
        middle_points.append(yearly_middle_point)
        amplitudes.append(yearly_amplitude)
    
    # Calculate the average middle point and amplitude over the past N years
    middle_point_historical = sum(middle_points) / len(middle_points)
    amplitude_historical_avg = sum(amplitudes) / len(amplitudes)
    
    # Calculate the mean middle point and amplitude for the year 2010 in future data
    future_2010 = future_data_original.sel(time=slice('2010-01', '2010-12'))
    middle_point_future = (future_2010.max() + future_2010.min()) / 2
    
    # Initialize the corrected future data array
    future_data_corrected = future_data_original.copy()
    
    # Adjustment based on the difference in middle points (Bias Adjustment)
    if use_bias_adjustment:
        adjustment = middle_point_historical - middle_point_future
        future_data_corrected = future_data_corrected + adjustment
    
    # Iterative rescaling to correct for negative values (moved before amplitude correction)
    if apply_iterative_rescaling_to_correct_for_negative_values:
        future_data_corrected = iterative_rescaling_v2(future_data_corrected, historical_data, middle_point_historical, minimum_future_ratio, start_year, last_year)
    
    # Initialize the amplitude corrected data array
    future_data_corrected_including_amplitude_effect_xarray = xr.DataArray(data=future_data_corrected * 0.0, dims=["time"], coords={"time": years_future})

    if apply_amplitude_correction:
        for year in range(2010, future_data_corrected.time.dt.year.max().values + 1):
            future_year = future_data_corrected.sel(time=slice(f'{year}-01', f'{year}-12'))

            Min_n_corrected_with_adjustment_only = future_year.min().values
            Max_n_corrected_with_adjustment_only = future_year.max().values
            Middle_point_n_corrected_with_adjustment_only = (Max_n_corrected_with_adjustment_only + Min_n_corrected_with_adjustment_only) / 2.0
            A_n_corrected_with_adjustment_only = Max_n_corrected_with_adjustment_only - Min_n_corrected_with_adjustment_only

            if year == 2010:
                if (middle_point_historical - (amplitude_historical_avg / 2.0)) < 0:
                    temp_minimum_value = minimum_future_ratio * historical_data.min().values
                    corrected_values = min_max_rescale(future_year, temp_minimum_value, middle_point_historical + (amplitude_historical_avg / 2.0))
                else:
                    corrected_values = min_max_rescale(future_year, middle_point_historical - (amplitude_historical_avg / 2.0), middle_point_historical + (amplitude_historical_avg / 2.0))
                future_data_corrected_including_amplitude_effect_xarray.loc[dict(time=future_year.time)] = corrected_values
                A_n_previous_corrected_with_adjustment_and_rescaling = amplitude_historical_avg
                A_n_previous_corrected_with_adjustment_only = A_n_corrected_with_adjustment_only
            else:
                if A_n_corrected_with_adjustment_only == 0:
                    Min_n = Middle_point_n_corrected_with_adjustment_only
                    Max_n = Middle_point_n_corrected_with_adjustment_only
                else:
                    Min_n = Middle_point_n_corrected_with_adjustment_only - (A_n_previous_corrected_with_adjustment_and_rescaling / 2) * (A_n_corrected_with_adjustment_only / A_n_previous_corrected_with_adjustment_only)
                    Max_n = Middle_point_n_corrected_with_adjustment_only + (A_n_previous_corrected_with_adjustment_and_rescaling / 2) * (A_n_corrected_with_adjustment_only / A_n_previous_corrected_with_adjustment_only)
                
                if Min_n < 0:
                    Min_n = minimum_future_ratio * historical_data.min().values
                    corrected_values = min_max_rescale(future_year, Min_n, Max_n)
                else:
                    corrected_values = min_max_rescale(future_year, Min_n, Max_n)

                future_data_corrected_including_amplitude_effect_xarray.loc[dict(time=future_year.time)] = corrected_values
                A_n_previous_corrected_with_adjustment_and_rescaling = Max_n - Min_n
                A_n_previous_corrected_with_adjustment_only = A_n_corrected_with_adjustment_only

    future_data_final = future_data_corrected_including_amplitude_effect_xarray if apply_amplitude_correction else future_data_corrected
    
    print("Value from the correction algo:")
    print(np.nansum(future_data_final))

    return future_data_final

def future_timeseries_correction_algorithm_v3(historical_data, future_data_original, years_future, sector,
                                           use_bias_adjustment=True, apply_amplitude_correction=True, 
                                           number_of_years_to_use_for_amplitude_reference_N=1, 
                                           apply_iterative_rescaling_to_correct_for_negative_values=True, 
                                           minimum_future_ratio=1.0):
    
    # Calculate the slice range for the past N years from the last available year in historical data
    last_year = historical_data.time.dt.year.max().values
    start_year = last_year - number_of_years_to_use_for_amplitude_reference_N + 1
    historical_period = historical_data.sel(time=slice(f'{start_year}-01', f'{last_year}-12'))
    
    # Calculate amplitudes for each year in the historical period
    amplitudes = []
    for year in range(start_year, last_year + 1):
        yearly_data = historical_period.sel(time=slice(f'{year}-01', f'{year}-12'))
        yearly_max = yearly_data.max(dim='time')
        yearly_min = yearly_data.min(dim='time')
        yearly_amplitude = yearly_max - yearly_min
        amplitudes.append(yearly_amplitude)
    
    # Calculate the average middle point and amplitude over the past N years
    hist_2010 = historical_period.sel(time=slice('2010-01', '2010-12'))
    middle_point_historical = (hist_2010.max() + hist_2010.min()) / 2.0
    amplitude_historical_avg = sum(amplitudes) / len(amplitudes)
    
    # Calculate the mean middle point and amplitude for the year 2010 in future data
    future_2010 = future_data_original.sel(time=slice('2010-01', '2010-12'))
    middle_point_future = (future_2010.max() + future_2010.min()) / 2
    
    # Initialize the corrected future data array
    future_data_corrected = future_data_original.copy()
    
    # Adjustment based on the difference in middle points (Bias Adjustment)
    if use_bias_adjustment:
        adjustment = middle_point_historical - middle_point_future
        future_data_corrected = future_data_corrected + adjustment
    
    # Iterative rescaling to correct for negative values (moved before amplitude correction)
    if apply_iterative_rescaling_to_correct_for_negative_values:
        future_data_corrected = iterative_rescaling_v2(future_data_corrected, historical_data, middle_point_historical, minimum_future_ratio, start_year, last_year)
    
    # Initialize the amplitude corrected data array
    future_data_corrected_including_amplitude_effect_xarray = xr.DataArray(data=future_data_corrected * 0.0, dims=["time"], coords={"time": years_future})

    if apply_amplitude_correction:
        for year in range(2010, future_data_corrected.time.dt.year.max().values + 1):
            future_year = future_data_corrected.sel(time=slice(f'{year}-01', f'{year}-12'))

            Min_n_corrected_with_adjustment_only = future_year.min().values
            Max_n_corrected_with_adjustment_only = future_year.max().values
            Middle_point_n_corrected_with_adjustment_only = (Max_n_corrected_with_adjustment_only + Min_n_corrected_with_adjustment_only) / 2.0
            A_n_corrected_with_adjustment_only = Max_n_corrected_with_adjustment_only - Min_n_corrected_with_adjustment_only

            if year == 2010:
                if (middle_point_historical - (amplitude_historical_avg / 2.0)) < 0:
                    temp_minimum_value = minimum_future_ratio * historical_data.min().values
                    corrected_values = min_max_rescale(future_year, temp_minimum_value, middle_point_historical + (amplitude_historical_avg / 2.0))
                else:
                    corrected_values = min_max_rescale(future_year, middle_point_historical - (amplitude_historical_avg / 2.0), middle_point_historical + (amplitude_historical_avg / 2.0))
                future_data_corrected_including_amplitude_effect_xarray.loc[dict(time=future_year.time)] = corrected_values
                A_n_previous_corrected_with_adjustment_and_rescaling = amplitude_historical_avg
                A_n_previous_corrected_with_adjustment_only = A_n_corrected_with_adjustment_only
            else:
                if A_n_corrected_with_adjustment_only == 0:
                    Min_n = Middle_point_n_corrected_with_adjustment_only
                    Max_n = Middle_point_n_corrected_with_adjustment_only
                else:
                    Min_n = Middle_point_n_corrected_with_adjustment_only - (A_n_previous_corrected_with_adjustment_and_rescaling / 2) * (A_n_corrected_with_adjustment_only / A_n_previous_corrected_with_adjustment_only)
                    Max_n = Middle_point_n_corrected_with_adjustment_only + (A_n_previous_corrected_with_adjustment_and_rescaling / 2) * (A_n_corrected_with_adjustment_only / A_n_previous_corrected_with_adjustment_only)
                
                if Min_n < 0:
                    Min_n = minimum_future_ratio * historical_data.min().values
                    corrected_values = min_max_rescale(future_year, Min_n, Max_n)
                else:
                    corrected_values = min_max_rescale(future_year, Min_n, Max_n)

                future_data_corrected_including_amplitude_effect_xarray.loc[dict(time=future_year.time)] = corrected_values
                A_n_previous_corrected_with_adjustment_and_rescaling = Max_n - Min_n
                A_n_previous_corrected_with_adjustment_only = A_n_corrected_with_adjustment_only
    
    future_data_final = future_data_corrected_including_amplitude_effect_xarray if apply_amplitude_correction else future_data_corrected
    
    # Final bias adjustment to ensure the minimum value is constrained
    if use_bias_adjustment and sector=="Irrigation":
        min_accepted_value = minimum_future_ratio * historical_data.min().values
        
        future_year_2010_min = future_data_final.sel(time=slice('2011-01', '2011-12')).min().values
        # print("Future min is: " + str(future_year_2010_min))
        hist_year_2010_min = historical_data.sel(time=slice('2010-01', '2010-12')).min().values
        # print("Hist min is: " + str(hist_year_2010_min))
        adjustment = hist_year_2010_min - future_year_2010_min
    
        # print("Final adjustment is: " + str(adjustment) + " km3/month")
        future_data_final = future_data_final + adjustment
        # Set all negative values to min_accepted_value
        future_data_final = xr.where(future_data_final < 0, min_accepted_value, future_data_final)
        # print("At the correction part")
        # print(future_data_final)
        
        print("Final bias adjustment was completed.")
    return future_data_final




def iterative_rescaling(future_data, historical_data, middle_point_historical, minimum_future_ratio, start_year, last_year):
    # Calculate minimum points for each year in the user selected interval in the historical period
    min_value_target = minimum_future_ratio * historical_data.min().values
    max_value_target = future_data.max().values
    max_value_original = future_data.max().values

    while True:
        if future_data.min().values >= 0:
            print("No negative values found, no rescaling needed.")
            break
        
        print("Negative values found, iterative rescaling applied.")
        # Apply min-max rescaling and convert the result back to xarray.DataArray
        rescaled_values = min_max_rescale(future_data, min_value_target, max_value_target)
        # future_data.values = rescaled_values
        
        
        # # Re-apply bias adjustment
        middle_point_future = np.nanmean(rescaled_values[0:12])
        # print(middle_point_future)
        historical_2010 = historical_data.sel(time=slice('2010-01', '2010-12'))
        middle_point_historical = (historical_2010.max() + historical_2010.min()) / 2
        # print(middle_point_historical)
        alignment = (middle_point_future - middle_point_historical)/ middle_point_historical
    
        # Exception handling: Skip to next country if historical middle point is zero
        if middle_point_historical == 0:
            print("Historical middle point is zero, skipping this country.")
            return future_data  # Exit the function and skip rescaling
        
        
        # Check for NaN
        # if relative_change_2011_2010_future_transition_year > relative_change_2012_2011_future:
        if alignment > 0.01:
            print("The timeseries are miss-aligned at the transition year by more than 1%.")
            print("Current miss-alignement is: " + str(np.round(alignment.item()*100, 2)) + "%")
            print("To achieve a smooth transition, we re-apply the min-max rescaling with maximum value adjusted " + str(np.round(0.01*alignment.item()*100/4 *100, 2)) + "%")
            
            max_value_target = max_value_target - max_value_original*(0.01*alignment.item()*100/4)
        else:
            print("Succesful correction of negative values while preserving trends has been completed.")
            future_data.values = rescaled_values
            break
    

    return future_data

def iterative_rescaling_v2(future_data, historical_data, middle_point_historical, minimum_future_ratio, start_year, last_year):
    # Calculate minimum points for each year in the user selected interval in the historical period
    min_value_target = minimum_future_ratio * historical_data.min().values
    max_value_target = future_data.max().values
    max_value_original = future_data.max().values

    while True:
        if future_data.min().values >= 0:
            print("No negative values found, no rescaling needed.")
            break
        
        print("Negative values found, iterative rescaling applied.")
        # Apply min-max rescaling and convert the result back to xarray.DataArray
        rescaled_values = min_max_rescale(future_data, min_value_target, max_value_target)        
        
        # Check the alignment
        middle_point_future = np.nanmean(rescaled_values[0:12])
        alignment = (middle_point_future - middle_point_historical)/ middle_point_historical
    
        # Exception handling: Skip to next country if historical middle point is zero
        if middle_point_historical == 0:
            print("Historical middle point is zero, skipping this country.")
            return future_data  # Exit the function and skip rescaling
        
        if alignment > 0.01:
            print("The timeseries are miss-aligned at the transition year by more than 1%.")

        
        if alignment > 0.01:
            print("Current miss-alignement is: " + str(np.round(alignment.item()*100, 2)) + "%")
            max_value_target -= np.clip(0.25 * alignment.item(), 0.005, 0.25) * max_value_target  
            print("To achieve a smooth transition, we re-apply the min-max rescaling with maximum value adjusted " + str(np.round(-np.clip(0.25 * alignment.item() , 0.005, 0.25)*100, 2)) + "%")
        else:
            print("Succesful correction of negative values while preserving trends has been completed.")
            future_data.values = rescaled_values
            break
    

    return future_data
  

    
def apply_the_correction_algorithm_at_country_and_US_state_level(countries_names, states_names, hist_countries_timeseries, future_countries_timeseries, hist_US_states_timeseries, future_US_states_timeseries, corrected_future_countries_timeseries, corrected_future_US_states_timeseries, years_future, use_bias_adjustment=True, apply_amplitude_correction=True, number_of_years_to_use_for_amplitude_reference_N=1, apply_iterative_rescaling_to_correct_for_negative_values=True, minimum_future_ratio=1.0):
    
    # Calculate total iterations
    total_iterations = len(hist_countries_timeseries) # number of countries in the dataset

    # Calculate the step size for each 25% increment
    step_size = total_iterations // 4  # Use integer division to ensure whole number steps
    
    for country_name in hist_countries_timeseries.country.values:
        if country_name != 4 and country_name != 177:
            print(countries_names[country_name])
            historical_data = hist_countries_timeseries.sel(country=country_name)
            future_data_original = future_countries_timeseries.sel(country=country_name)
            future_data_corrected = future_timeseries_correction_algorithm(historical_data, future_data_original, years_future, use_bias_adjustment, apply_amplitude_correction, number_of_years_to_use_for_amplitude_reference_N, apply_iterative_rescaling_to_correct_for_negative_values, minimum_future_ratio)
            corrected_future_countries_timeseries[country_name] = future_data_corrected
        elif country_name == 4:
            for state in hist_US_states_timeseries.US_states.values:
                if state != 51:
                    print(states_names[state])
                    historical_data = hist_US_states_timeseries.sel(US_states=state)
                    future_data_original = future_US_states_timeseries.sel(US_states=state)
                    future_data_corrected = future_timeseries_correction_algorithm(historical_data, future_data_original, years_future, use_bias_adjustment, apply_amplitude_correction, number_of_years_to_use_for_amplitude_reference_N, apply_iterative_rescaling_to_correct_for_negative_values, minimum_future_ratio)
                    corrected_future_US_states_timeseries[state] = future_data_corrected
                else:
                    print("No correction applied to gridcells outside the regionmask US states boundaries (state code 51).")
        else:
            print("No correction applied to gridcells outside the regionmask country boundaries (country code 177).")
        
        # Print a progress update
        if country_name % step_size == 0 or country_name == total_iterations:
            percent_complete = (country_name / total_iterations) * 100
            print(f"Calculation completed at {int(percent_complete)}%")

    print("Future countries and US states timeseries are corrected. The updated values can be found in the provided input xarray.DataArray's.")
    

    
def apply_the_correction_algorithm_at_country_and_US_state_level_v2(countries_names, states_names, hist_countries_timeseries, future_countries_timeseries, hist_US_states_timeseries, future_US_states_timeseries, corrected_future_countries_timeseries, corrected_future_US_states_timeseries, years_future, use_bias_adjustment=True, apply_amplitude_correction=True, number_of_years_to_use_for_amplitude_reference_N=1, apply_iterative_rescaling_to_correct_for_negative_values=True, minimum_future_ratio=1.0):
    
    # Calculate total iterations
    total_iterations = len(hist_countries_timeseries) # number of countries in the dataset

    # Calculate the step size for each 25% increment
    step_size = total_iterations // 4  # Use integer division to ensure whole number steps
    
    for country_name in hist_countries_timeseries.country.values:
        if country_name != 4 and country_name != 177:
            print(countries_names[country_name])
            historical_data = hist_countries_timeseries.sel(country=country_name)
            future_data_original = future_countries_timeseries.sel(country=country_name)
            future_data_corrected = future_timeseries_correction_algorithm_v2(historical_data, future_data_original, years_future, use_bias_adjustment, apply_amplitude_correction, number_of_years_to_use_for_amplitude_reference_N, apply_iterative_rescaling_to_correct_for_negative_values, minimum_future_ratio)
            corrected_future_countries_timeseries[country_name] = future_data_corrected
        elif country_name == 4:
            for state in hist_US_states_timeseries.US_states.values:
                if state != 51:
                    print(states_names[state])
                    historical_data = hist_US_states_timeseries.sel(US_states=state)
                    future_data_original = future_US_states_timeseries.sel(US_states=state)
                    future_data_corrected = future_timeseries_correction_algorithm_v2(historical_data, future_data_original, years_future, use_bias_adjustment, apply_amplitude_correction, number_of_years_to_use_for_amplitude_reference_N, apply_iterative_rescaling_to_correct_for_negative_values, minimum_future_ratio)
                    corrected_future_US_states_timeseries[state] = future_data_corrected
                else:
                    print("No correction applied to gridcells outside the regionmask US states boundaries (state code 51).")
        else:
            print("No correction applied to gridcells outside the regionmask country boundaries (country code 177).")
        
        # Print a progress update
        if country_name % step_size == 0 or country_name == total_iterations:
            percent_complete = (country_name / total_iterations) * 100
            print(f"Calculation completed at {int(percent_complete)}%")

    print("Future countries and US states timeseries are corrected. The updated values can be found in the provided input xarray.DataArray's.")
    
    
def apply_the_correction_algorithm_at_country_and_US_state_level_v3(countries_names, states_names, hist_countries_timeseries, future_countries_timeseries, hist_US_states_timeseries, future_US_states_timeseries, corrected_future_countries_timeseries, corrected_future_US_states_timeseries, years_future, sector, use_bias_adjustment=True, apply_amplitude_correction=True, number_of_years_to_use_for_amplitude_reference_N=1, apply_iterative_rescaling_to_correct_for_negative_values=True, minimum_future_ratio=1.0):
    
    # Calculate total iterations
    total_iterations = len(hist_countries_timeseries) # number of countries in the dataset

    # Calculate the step size for each 25% increment
    step_size = total_iterations // 4  # Use integer division to ensure whole number steps
    
    for country_name in hist_countries_timeseries.country.values:
        if country_name != 4 and country_name != 177:
            print(countries_names[country_name])
            historical_data = hist_countries_timeseries.sel(country=country_name)
            future_data_original = future_countries_timeseries.sel(country=country_name)
            future_data_corrected = future_timeseries_correction_algorithm_v3(historical_data, future_data_original, years_future, sector, use_bias_adjustment, apply_amplitude_correction, number_of_years_to_use_for_amplitude_reference_N, apply_iterative_rescaling_to_correct_for_negative_values, minimum_future_ratio)
            corrected_future_countries_timeseries[country_name] = future_data_corrected
        elif country_name == 4:
            for state in hist_US_states_timeseries.US_states.values:
                if state != 51:
                    print(states_names[state])
                    historical_data = hist_US_states_timeseries.sel(US_states=state)
                    future_data_original = future_US_states_timeseries.sel(US_states=state)
                    future_data_corrected = future_timeseries_correction_algorithm_v3(historical_data, future_data_original, years_future, sector, use_bias_adjustment, apply_amplitude_correction, number_of_years_to_use_for_amplitude_reference_N, apply_iterative_rescaling_to_correct_for_negative_values, minimum_future_ratio)
                    corrected_future_US_states_timeseries[state] = future_data_corrected
                else:
                    print("No correction applied to gridcells outside the regionmask US states boundaries (state code 51).")
        else:
            print("No correction applied to gridcells outside the regionmask country boundaries (country code 177).")
        
        # Print a progress update
        if country_name % step_size == 0 or country_name == total_iterations:
            percent_complete = (country_name / total_iterations) * 100
            print(f"Calculation completed at {int(percent_complete)}%")

    print("Future countries and US states timeseries are corrected. The updated values can be found in the provided input xarray.DataArray's.")
    
    
def downscale_future_corrected_national_and_US_states_timeseries_to_gridded_format_and_save_it(future_data_gridded_format, path_to_save_corrected_gridded_data, mask_countries, states_index, mask_US, countries_index, corrected_future_countries_timeseries, corrected_future_US_states_timeseries, variable_name, prefix_for_output):
    # Initialize a copy of the original dataset to hold corrected values
    corrected_gridded_data = future_data_gridded_format[variable_name].copy(deep=True) * 0.0

    # Iterate over each month and country
    for time_idx in range(future_data_gridded_format[variable_name].sizes['time']):
        # Temporary arrays to store intermediate calculations
        downscaling_coefficients = future_data_gridded_format[variable_name].isel(time=time_idx).copy(deep=True) * 0.0
        mask_with_total_corrected = future_data_gridded_format[variable_name].isel(time=time_idx).copy(deep=True) * 0.0
        array_ones = future_data_gridded_format[variable_name].isel(time=time_idx).copy(deep=True) * 0.0 + 1.0

        for country_idx in countries_index:
            if country_idx == 177:
                # The country code 177 corresponds to gridcells which are not part of any country
                # We still want to include them in the summation process, not to loose this information
                # But since we didn't applied any correction to code 177, it is enough to keep the original values:
                original_total = future_data_gridded_format[variable_name][time_idx].where(mask_countries == country_idx).sum()
                downscaling_coefficients += (future_data_gridded_format[variable_name][time_idx].where(mask_countries == country_idx) / original_total).fillna(0)
                mask_with_total_corrected += (array_ones.where(mask_countries == country_idx) * original_total).fillna(0)

            elif country_idx == 4:
                for state in states_index:
                    if state == 51:
                        # ignore code 51 (not a US state)
                        # it will contain the remaining countries too, so to avoid double counting we should not add it to the summation
                        continue
                    else:
                        # Select the corrected total withdrawal for the given US state and month
                        corrected_total = corrected_future_US_states_timeseries[state, time_idx]
                        # Calculate the total withdrawal from the original data for the given US state and month
                        original_total = future_data_gridded_format[variable_name][time_idx].where(mask_US == state).sum()
                        # Calculate downscaling coefficients for each grid cell
                        downscaling_coefficients += (future_data_gridded_format[variable_name][time_idx].where(mask_US == state) / original_total).fillna(0)
                        # Distribute the corrected total uniformly across all grid cells of the country
                        mask_with_total_corrected += (array_ones.where(mask_US == state) * corrected_total).fillna(0)
            else:
                # Same procedure as for US states, but for other countries 
                corrected_total = corrected_future_countries_timeseries[country_idx, time_idx]
                original_total = future_data_gridded_format[variable_name][time_idx].where(mask_countries == country_idx).sum()
                downscaling_coefficients += (future_data_gridded_format[variable_name][time_idx].where(mask_countries == country_idx) / original_total).fillna(0)
                mask_with_total_corrected += (array_ones.where(mask_countries == country_idx) * corrected_total).fillna(0)

        # Apply downscaling coefficients to the corrected totals to obtain the downscaled data
        corrected_gridded_monthly_data = mask_with_total_corrected * downscaling_coefficients
        # Assign the corrected, downscaled data back to the corresponding month in the dataset
        corrected_gridded_data[time_idx] = corrected_gridded_data[time_idx] + corrected_gridded_monthly_data
        # Return 0 population values into NaNs:
        corrected_gridded_data[time_idx] = corrected_gridded_data[time_idx].where(corrected_gridded_data[time_idx] != 0)
        
        if (time_idx + 1) % 12 == 0:
            year = 2010 + time_idx // 12
            if year % 10 == 0:
                print("Year completed: " + str(year))

    for year in range(2010, 2101):
        yearly_data = corrected_gridded_data.sel(time=slice(f"{year}-01", f"{year}-12"))
        file_name = f"{prefix_for_output}_m3_per_day_with_corrected_national_time_series_{year}.nc"
        file_path = f"{path_to_save_corrected_gridded_data}/{file_name}"
        yearly_data.to_netcdf(file_path)
        print(f"Saved {file_name}")
        