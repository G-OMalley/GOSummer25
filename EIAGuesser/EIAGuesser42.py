import pandas as pd
import os
import re

# --- Define Paths ---
# The folder where your input CSV files are located.
input_path = r'C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\INFO'
# The folder where you want to save the output file.
output_path = r'C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\EIAGuesser\output'
# The full path to the input weather file.
weather_file_path = os.path.join(input_path, 'WEATHER.csv')
# The full path to the PriceAdmin file for region mapping.
price_admin_path = os.path.join(input_path, 'PriceAdmin.csv')
# The full path to the Platts Fundamentals file.
platts_fundy_path = os.path.join(input_path, 'PlattsCONUSFundamentalsHIST.csv')
# The full path to the Platts Power Fundamentals file.
platts_power_fundy_path = os.path.join(input_path, 'PlattsPowerFundy.csv')
# The full path to the Criterion Fundamentals file.
fundy_path = os.path.join(input_path, 'Fundy.csv')
# The full path to the Criterion Extra Fundamentals file.
critextra_path = os.path.join(input_path, 'CriterionExtra.csv')
# The full path to the Criterion Storage Change file.
storagechange_path = os.path.join(input_path, 'CriterionStorageChange.csv')
# The full path to the locations list for mapping.
locs_list_path = os.path.join(input_path, 'locs_list.csv')

# --- Helper Function for Robust Column Cleaning ---
def clean_columns(df):
    """
    Cleans all column names in a DataFrame by converting to lowercase,
    stripping whitespace, and replacing all non-alphanumeric characters with underscores.
    """
    new_columns = []
    for col in df.columns:
        # Convert to lowercase and strip whitespace
        clean_col = col.strip().lower()
        # Replace spaces and other problematic characters with a single underscore
        clean_col = re.sub(r'[^a-z0-9]+', '_', clean_col)
        # Remove any leading/trailing underscores that might result
        clean_col = clean_col.strip('_')
        new_columns.append(clean_col)
    df.columns = new_columns
    return df


# --- Create Output Directory ---
# Create the output directory if it doesn't already exist.
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"Created output directory: {output_path}")


# --- Load Data and Apply Universal Cleaning ---
try:
    df_weather = clean_columns(pd.read_csv(weather_file_path))
    print(f"Successfully loaded and cleaned {weather_file_path}")
    df_price_admin = clean_columns(pd.read_csv(price_admin_path))
    print(f"Successfully loaded and cleaned {price_admin_path}")
    df_platts = clean_columns(pd.read_csv(platts_fundy_path))
    print(f"Successfully loaded and cleaned {platts_fundy_path}")
    df_power = clean_columns(pd.read_csv(platts_power_fundy_path))
    print(f"Successfully loaded and cleaned {platts_power_fundy_path}")
    df_fundy = clean_columns(pd.read_csv(fundy_path))
    print(f"Successfully loaded and cleaned {fundy_path}")
    df_critextra = clean_columns(pd.read_csv(critextra_path))
    print(f"Successfully loaded and cleaned {critextra_path}")
    df_storagechange = clean_columns(pd.read_csv(storagechange_path))
    print(f"Successfully loaded and cleaned {storagechange_path}")
    df_locs = clean_columns(pd.read_csv(locs_list_path))
    print(f"Successfully loaded and cleaned {locs_list_path}")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()


# --- Create City to EIA Region Map ---
# Standardize the data within the 'region' column to ensure the map works correctly.
df_price_admin['region'] = df_price_admin['region'].str.lower().str.replace(' ', '_')
mapper_df = df_price_admin[['city_symbol', 'region']].drop_duplicates('city_symbol').copy()
priceadmin_to_eia_map = {
    'northeast': 'East', 'southeast': 'East', 'midwest': 'Midwest',
    'south_central': 'SouthCentral', 'rockies': 'Mountain', 'west': 'Pacific'
}
mapper_df['eia_region'] = mapper_df['region'].map(priceadmin_to_eia_map)
city_to_eia_region_map = pd.Series(mapper_df.eia_region.values, index=mapper_df.city_symbol).to_dict()
print("\nCreated city-to-EIA-region map from PriceAdmin.csv")


# --- Process Weather Data ---
df_weather.rename(columns={'min_temp': 'mintemp', 'max_temp': 'maxtemp', 'avg_temp': 'avgtemp',
                           'min_feels_like': 'minfeelslike', 'max_feels_like': 'maxfeelslike',
                           'max_surface_wind': 'maxsurfacewind', '10yr_min_temp': '10yrmintemp',
                           '10yr_max_temp': '10yrmaxtemp', '10yr_avg_temp': '10yravgtemp'}, inplace=True)
df_weather['date'] = pd.to_datetime(df_weather['date'])
df_weather.set_index(['date', 'city_symbol'], inplace=True)
df_pivot = df_weather.unstack(level='city_symbol')
df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]
cities = df_weather.index.get_level_values('city_symbol').unique()

all_city_data = []
for city in cities:
    region = city_to_eia_region_map.get(city, 'unknown')
    city_data = {}
    for metric in ['mintemp', 'maxtemp', 'avgtemp', 'cdd', 'hdd', 'minfeelslike', 'maxfeelslike', 'maxsurfacewind', '10yrmintemp', '10yrmaxtemp', '10yravgtemp']:
        new_col_name = f"{region}_weather_{city}_{metric}"
        pivoted_col_name = f"{metric}_{city}"
        if pivoted_col_name in df_pivot.columns:
            city_data[new_col_name] = df_pivot[pivoted_col_name]
    all_city_data.append(pd.DataFrame(city_data))

processed_weather_df = pd.concat(all_city_data, axis=1)

for city in cities:
    region = city_to_eia_region_map.get(city, 'unknown')
    mintemp_col = f'{region}_weather_{city}_mintemp'
    yrmintemp_col = f'{region}_weather_{city}_10yrmintemp'
    maxtemp_col = f'{region}_weather_{city}_maxtemp'
    yrmaxtemp_col = f'{region}_weather_{city}_10yrmaxtemp'
    avgtemp_col = f'{region}_weather_{city}_avgtemp'
    yravgtemp_col = f'{region}_weather_{city}_10yravgtemp'
    if all(c in processed_weather_df.columns for c in [mintemp_col, yrmintemp_col, maxtemp_col, yrmaxtemp_col, avgtemp_col, yravgtemp_col]):
        processed_weather_df[f'{region}_weather_{city}_mintemp_vs_10yr'] = processed_weather_df[mintemp_col] - processed_weather_df[yrmintemp_col]
        processed_weather_df[f'{region}_weather_{city}_maxtemp_vs_10yr'] = processed_weather_df[maxtemp_col] - processed_weather_df[yrmaxtemp_col]
        processed_weather_df[f'{region}_weather_{city}_avgtemp_vs_10yr'] = processed_weather_df[avgtemp_col] - processed_weather_df[yravgtemp_col]

print("\nProcessed weather data.")


# --- Process Platts Gas Fundamentals Data ---
df_platts.rename(columns={'gasdate': 'date'}, inplace=True)
df_platts['date'] = pd.to_datetime(df_platts['date'])
df_platts.set_index('date', inplace=True)
df_platts.columns = [f"conus_plattsfundy_{col}" for col in df_platts.columns]
print("Processed Platts Gas Fundamentals data.")


# --- Process Platts Power Fundamentals Data ---
POWER_ITEM_TO_REGION_MAP = {
    'AESO': ['Mountain'], 'BPA': ['Pacific'], 'CAISO': ['Pacific'],
    'ERCOT': ['SouthCentral'], 'IESO': ['Midwest', 'East'],
    'ISONE': ['East'], 'MISO': ['Midwest', 'SouthCentral'],
    'NYISO': ['East'], 'PJM': ['East'], 'SPP': ['Midwest', 'SouthCentral']
}
df_power.rename(columns={'effective_date': 'date'}, inplace=True)
df_power['date'] = pd.to_datetime(df_power['date'])
df_power['iso'] = df_power['item'].apply(lambda x: x.split('_')[0].upper())
df_power['region'] = df_power['iso'].map(POWER_ITEM_TO_REGION_MAP)
df_power['clean_item'] = df_power['item'].str.replace(r'[^a-z0-9]+', '_', regex=True).str.strip('_')
df_power = df_power.explode('region')
df_power['final_col_name'] = df_power['region'] + '_powerfundy_' + df_power['clean_item']
processed_power_df = df_power.pivot_table(index='date', columns='final_col_name', values='value', aggfunc='sum')
print("Processed Platts Power Fundamentals data.")


# --- Process Criterion Fundamentals Data ---
fundy_region_map = {
    'conus': 'CONUS48', 'lower_48': 'CONUS48', 'gulf_of_mexico': 'SouthCentral',
    'northeast': 'East', 'southcentral': 'SouthCentral', 'south_central': 'SouthCentral',
    'southeast': 'East', 'california': 'Pacific', 'midwest': 'Midwest', 'rockies': 'Mountain',
    'west': 'Pacific'
}
df_fundy.rename(columns={'gas_day': 'date'}, inplace=True)
df_fundy['date'] = pd.to_datetime(df_fundy['date'])
df_fundy['mapped_region'] = df_fundy['region'].map(fundy_region_map)
df_fundy['clean_item'] = df_fundy['item'].str.replace(r'[^a-zA-Z0-9]+', '', regex=True)
df_fundy['final_col_name'] = df_fundy['mapped_region'] + '_fundy_' + df_fundy['clean_item']
processed_fundy_df = df_fundy.pivot_table(index='date', columns='final_col_name', values='value', aggfunc='sum')
print("Processed Criterion Fundamentals data.")


# --- Process Criterion Extra Data ---
df_critextra.rename(columns={'gas_day': 'date'}, inplace=True)
df_critextra['date'] = pd.to_datetime(df_critextra['date'])
critextra_item_to_region_map = {
    'conus_storage': 'CONUS48', 'total_demand_lower_48': 'CONUS48',
    'total_demand_california': 'Pacific', 'total_demand_midwest': 'Midwest',
    'total_demand_northeast': 'East', 'total_demand_rockies': 'Mountain',
    'total_demand_south_central': 'SouthCentral', 'total_demand_southeast': 'East',
    'total_demand_pacific_northwest': 'Pacific'
}
critextra_item_to_metric_map = {
    'conus_storage': 'storage', 'total_demand_lower_48': 'totaldemand',
    'total_demand_california': 'totaldemand', 'total_demand_midwest': 'totaldemand',
    'total_demand_northeast': 'totaldemand', 'total_demand_rockies': 'totaldemand',
    'total_demand_south_central': 'totaldemand', 'total_demand_southeast': 'totaldemand',
    'total_demand_pacific_northwest': 'totaldemand'
}
df_critextra['mapped_region'] = df_critextra['item'].map(critextra_item_to_region_map)
df_critextra['clean_metric'] = df_critextra['item'].map(critextra_item_to_metric_map)
df_critextra['final_col_name'] = df_critextra['mapped_region'] + '_critextra_' + df_critextra['clean_metric']
processed_critextra_df = df_critextra.pivot_table(index='date', columns='final_col_name', values='value', aggfunc='sum')
print("Processed Criterion Extra data.")


# --- Process Criterion Storage Change Data ---
state_to_region_map = {
    'alabama': 'SouthCentral', 'arizona': 'Mountain', 'arkansas': 'SouthCentral',
    'california': 'Pacific', 'colorado': 'Mountain', 'florida': 'East',
    'illinois': 'Midwest', 'indiana': 'Midwest', 'iowa': 'Midwest',
    'kansas': 'SouthCentral', 'kentucky': 'Midwest', 'louisiana': 'SouthCentral',
    'maryland': 'East', 'michigan': 'Midwest', 'mississippi': 'SouthCentral',
    'missouri': 'Midwest', 'montana': 'Mountain', 'nebraska': 'Midwest',
    'new_mexico': 'SouthCentral', 'new_york': 'East', 'north_dakota': 'Mountain',
    'ohio': 'Midwest', 'oklahoma': 'SouthCentral', 'oregon': 'Pacific',
    'pennsylvania': 'East', 'texas': 'SouthCentral', 'utah': 'Mountain',
    'virginia': 'East', 'washington': 'Pacific', 'west_virginia': 'East',
    'wyoming': 'Mountain'
}
df_storagechange.rename(columns={'eff_gas_day': 'date'}, inplace=True)
storage_to_state_map = pd.Series(df_locs.state_name.values, index=df_locs.storage_name).to_dict()
df_storagechange['state'] = df_storagechange['storage_name'].map(storage_to_state_map)
df_storagechange['region'] = df_storagechange['state'].map(state_to_region_map)
df_storagechange['clean_storage_name'] = df_storagechange['storage_name'].str.replace(r'[^a-zA-Z0-9]+', '', regex=True)
df_storagechange['final_col_name'] = df_storagechange['region'] + '_storagechange_' + df_storagechange['clean_storage_name']
df_storagechange['date'] = pd.to_datetime(df_storagechange['date'])

# Programmatically find the value column to pivot on, making the script more robust.
known_id_cols = {'date', 'storage_name', 'state', 'region', 'clean_storage_name', 'final_col_name'}
value_col = list(set(df_storagechange.columns) - known_id_cols)[0]
print(f"Dynamically identified value column for pivot: '{value_col}'")
processed_storagechange_df = df_storagechange.pivot_table(index='date', columns='final_col_name', values=value_col, aggfunc='sum')
print("Processed Criterion Storage Change data.")


# --- Combine DataFrames ---
# Merge the processed dataframes on their common 'Date' index.
final_df = processed_weather_df.merge(df_platts, left_index=True, right_index=True, how='left')
final_df = final_df.merge(processed_power_df, left_index=True, right_index=True, how='left')
final_df = final_df.merge(processed_fundy_df, left_index=True, right_index=True, how='left')
final_df = final_df.merge(processed_critextra_df, left_index=True, right_index=True, how='left')
final_df = final_df.merge(processed_storagechange_df, left_index=True, right_index=True, how='left')
print("\nMerged all data sources.")


# --- Final Processing and Saving ---
# To get a de-fragmented frame, make a copy.
final_df = final_df.copy()

# Filter the DataFrame to include only data from 2018 onwards.
final_df = final_df[final_df.index >= '2018-01-01']
print("\nFiltered data to include dates from 2018-01-01 onwards.")

# For maximum compatibility with different models, convert all column headers to lowercase.
final_df.columns = [col.lower() for col in final_df.columns]
print("\nConverted all column headers to lowercase for model compatibility.")

# Define the full path for the output file.
output_file = os.path.join(output_path, 'processed_daily_data.csv')

# Save the final, processed DataFrame to a CSV file.
final_df.to_csv(output_file)

print(f"\nProcessed daily data saved to: {output_file}")
print("\nFirst 5 rows of the final data:")
print(final_df.head())
