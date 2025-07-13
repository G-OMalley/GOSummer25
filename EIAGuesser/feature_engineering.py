# C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\EIAGuesser\Feature_engineering.py
# -*- coding: utf-8 -*-
"""
This script performs feature engineering for the EIA Guesser project.
It reads raw data from multiple sources, cleans and transforms the data,
creates a rich feature set based on regional and granular metrics,
and saves the final collated dataset for model training.

VERSION 5: LAGGED & CALENDAR FEATURES
This version introduces critical lagged features to prevent data leakage and
ensure the model only trains on information that would be available at the
time of a real-world forecast. It also adds calendar-aware features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from pandas.tseries.holiday import USFederalHolidayCalendar

# --- Configuration & Path Setup ---
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

INFO_DIR = SCRIPT_DIR.parent / 'INFO'
OUTPUT_DIR = SCRIPT_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Mappings & Constants ---
REGIONS = ['East', 'Midwest', 'Mountain', 'Pacific', 'South Central']

CITY_TO_REGION_MAP = {
    'Atlanta': 'East', 'Boston': 'East', 'Buffalo': 'East',
    'Washington': 'East', 'JFK': 'East', 'Pittsburgh': 'East',
    'Ral-Durham': 'East', 'Tampa': 'East',
    'Chicago OHare': 'Midwest', 'Detroit': 'Midwest',
    'Denver': 'Mountain',
    'Los Angeles': 'Pacific', 'Seattle': 'Pacific', 'San Francisco': 'Pacific',
    'Houston IAH': 'South Central', 'Little Rock': 'South Central',
    'New Orleans': 'South Central', 'Ok. City': 'South Central'
}

LNG_ITEM_TO_REGION_MAP = {
    'Calcasieu Pass LNG Feed Gas': 'South Central',
    'Cameron LNG Feed Gas': 'South Central',
    'Corpus Christi LNG Feed Gas': 'South Central',
    'Cove Point LNG Feed Gas': 'East',
    'Elba Island LNG Feed Gas': 'South Central',
    'Freeport LNG Feed Gas': 'South Central',
    'Sabine Pass LNG Feed Gas': 'South Central',
    'US LNG Exports - Plaquemines LNG': 'South Central'
}

POWER_ITEM_TO_REGION_MAP = {
    'AESO': ['Mountain'], 'BPA': ['Pacific'], 'CAISO': ['Pacific'],
    'ERCOT': ['South Central'], 'IESO': ['Midwest', 'East'],
    'ISONE': ['East'], 'MISO': ['Midwest', 'South Central'],
    'NYISO': ['East'], 'PJM': ['East'], 'SPP': ['Midwest', 'South Central']
}

CRITERION_EXTRA_MAP = {
    'CONUS - STORAGE': 'CONUS_Criterion_Storage',
    'Total Demand - California': 'Pacific_Criterion_Demand_CA',
    'Total Demand - Lower 48': 'CONUS_Criterion_Total_Demand',
    'Total Demand - Midwest': 'Midwest_Criterion_Total_Demand',
    'Total Demand - Northeast': 'East_Criterion_Total_Demand',
    'Total Demand - Pacific Northwest': 'Pacific_Criterion_Demand_PNW',
    'Total Demand - Rockies': 'Mountain_Criterion_Total_Demand',
    'Total Demand - Rockies - SW': 'Mountain_Criterion_Demand_SW',
    'Total Demand - Rockies - Upper': 'Mountain_Criterion_Demand_Upper',
    'Total Demand - South Central': 'South_Central_Criterion_Total_Demand',
    'Total Demand - Southeast - Florida': 'East_Criterion_Demand_SE_FL',
    'Total Demand - Southeast - Other': 'East_Criterion_Demand_SE_Other'
}

# --- Helper Functions ---

def sanitize_name(name):
    """Cleans a string to be used as a DataFrame column name."""
    return name.strip().replace(' ', '_').replace('-', '_').replace('.', '').replace('(', '').replace(')', '')

# --- Data Processing Functions ---

def process_weather_data(info_dir):
    print("Processing Weather Data...")
    weather_file = info_dir / 'WEATHER.csv'
    if not weather_file.exists(): return pd.DataFrame()
    df_weather = pd.read_csv(weather_file)
    df_weather['Date'] = pd.to_datetime(df_weather['Date'])
    df_weather['Region'] = df_weather['City Title'].map(CITY_TO_REGION_MAP)
    df_weather.dropna(subset=['Region'], inplace=True)
    agg_dict = {'CDD': 'sum', 'HDD': 'sum'}
    df_regional_agg = df_weather.groupby(['Date', 'Region']).agg(agg_dict).reset_index()
    df_regional_pivot = df_regional_agg.pivot(index='Date', columns='Region', values=list(agg_dict.keys()))
    df_regional_pivot.columns = [f'{region}_Weather_{stat}' for stat, region in df_regional_pivot.columns]
    df_city_pivots = []
    for region, df_group in df_weather.groupby('Region'):
        df_pivot = df_group.pivot(index='Date', columns='City Title', values=['Avg Temp', 'CDD', 'HDD'])
        df_pivot.columns = [f"{region}_Weather_City_{sanitize_name(city)}_{stat}" for stat, city in df_pivot.columns]
        df_city_pivots.append(df_pivot)
    df_city_granular = pd.concat(df_city_pivots, axis=1)
    df_final = df_regional_pivot.join(df_city_granular, how='outer')
    print("  - Weather data processing complete.")
    return df_final

def process_platts_conus_data(info_dir):
    print("Processing Platts CONUS Data...")
    conus_file = info_dir / 'PlattsCONUSFundamentalsHIST.csv'
    if not conus_file.exists(): return pd.DataFrame()
    df_conus = pd.read_csv(conus_file)
    df_conus['GasDate'] = pd.to_datetime(df_conus['GasDate'])
    df_conus.set_index('GasDate', inplace=True)
    df_conus.columns = [f"CONUS_Platts_{sanitize_name(col)}" for col in df_conus.columns]
    print("  - Platts CONUS data processing complete.")
    return df_conus

def process_power_data(info_dir):
    print("Processing Platts Power Data...")
    power_file = info_dir / 'PlattsPowerFundy.csv'
    if not power_file.exists(): return pd.DataFrame()
    df_power_long = pd.read_csv(power_file)
    df_power_long['Date'] = pd.to_datetime(df_power_long['Date'])
    df_power_long.dropna(subset=['Item', 'Value'], inplace=True)
    def get_iso_prefix(item_name):
        return item_name.strip().split(' - ')[0]
    df_power_long['ISO'] = df_power_long['Item'].apply(get_iso_prefix)
    df_power_long['Region'] = df_power_long['ISO'].map(POWER_ITEM_TO_REGION_MAP)
    df_power_long.dropna(subset=['Region'], inplace=True)
    df_exploded = df_power_long.explode('Region')
    df_exploded['Feature_Name'] = df_exploded.apply(lambda row: f"{row['Region']}_Power_{sanitize_name(row['Item'])}", axis=1)
    df_power_wide = df_exploded.pivot_table(index='Date', columns='Feature_Name', values='Value', aggfunc='mean')
    print("  - Platts Power data processing complete.")
    return df_power_wide

def process_nuclear_data(info_dir):
    print("Processing Nuclear Data...")
    nuclear_file = info_dir / 'CriterionNuclearHist.csv'
    if not nuclear_file.exists(): return pd.DataFrame()
    df_nuclear = pd.read_csv(nuclear_file)
    df_nuclear['Date'] = pd.to_datetime(df_nuclear['Date'])
    df_nuclear.dropna(subset=['EIA Region', 'Value'], inplace=True)
    df_regional = df_nuclear.groupby(['Date', 'EIA Region'])['Value'].sum().reset_index()
    df_pivot = df_regional.pivot(index='Date', columns='EIA Region', values='Value')
    df_pivot.columns = [f"{sanitize_name(col)}_Nuclear_Total" for col in df_pivot.columns]
    print("  - Nuclear data processing complete.")
    return df_pivot.fillna(0)

def process_lng_data(info_dir):
    print("Processing LNG Data...")
    lng_file = info_dir / 'CriterionLNGHist.csv'
    if not lng_file.exists(): return pd.DataFrame()
    df_lng_long = pd.read_csv(lng_file)
    df_lng_long['Date'] = pd.to_datetime(df_lng_long['Date'])
    df_lng_long['Region'] = df_lng_long['Item'].map(LNG_ITEM_TO_REGION_MAP)
    df_lng_long.dropna(subset=['Region'], inplace=True)
    df_lng_long['Feature_Name'] = df_lng_long.apply(lambda row: f"{row['Region']}_LNG_{sanitize_name(row['Item'])}", axis=1)
    df_lng_granular = df_lng_long.pivot_table(index='Date', columns='Feature_Name', values='Value', aggfunc='mean')
    df_regional_lng = df_lng_long.groupby(['Date', 'Region'])['Value'].sum().reset_index()
    df_lng_regional = df_regional_lng.pivot(index='Date', columns='Region', values='Value')
    df_lng_regional.columns = [f'{sanitize_name(col)}_LNG_Feedgas_Total' for col in df_lng_regional.columns]
    df_final = df_lng_regional.join(df_lng_granular, how='outer').fillna(0)
    print("  - LNG data processing complete.")
    return df_final

def process_criterion_extra_data(info_dir):
    print("Processing Criterion Extra Data...")
    extra_file = info_dir / 'CriterionExtra.csv'
    if not extra_file.exists(): return pd.DataFrame()
    df_extra_long = pd.read_csv(extra_file)
    df_extra_long['Date'] = pd.to_datetime(df_extra_long['Date'])
    df_filtered = df_extra_long[df_extra_long['item'].isin(CRITERION_EXTRA_MAP.keys())].copy()
    if df_filtered.empty: return pd.DataFrame()
    df_filtered['Feature_Name'] = df_filtered['item'].map(CRITERION_EXTRA_MAP)
    df_extra_wide = df_filtered.pivot_table(index='Date', columns='Feature_Name', values='value', aggfunc='mean')
    print("  - Criterion Extra data processing complete.")
    return df_extra_wide

def process_fundy_data(info_dir):
    print("Processing Fundy Data...")
    fundy_file = info_dir / 'Fundy.csv'
    if not fundy_file.exists(): return pd.DataFrame()
    df_fundy_long = pd.read_csv(fundy_file)
    df_fundy_long['Date'] = pd.to_datetime(df_fundy_long['Date'])
    df_fundy_wide = df_fundy_long.pivot_table(index='Date', columns='item', values='value', aggfunc='mean')
    df_fundy_wide.columns = [f"Fundy_{sanitize_name(col)}" for col in df_fundy_wide.columns]
    print("  - Fundy data processing complete.")
    return df_fundy_wide

def get_storage_to_region_map():
    print("  - Using curated, hardcoded storage-to-region map for reliability.")
    storage_map = {
        'ANR Pipeline': 'South Central', 'ANR Storage': 'Midwest', 'Arcadia Gas Storage': 'South Central', 'Avon Storage': 'Midwest',
        'Bay Gas Storage Company': 'South Central', 'Bear Creek Storage Company': 'South Central', 'Bistineau': 'South Central',
        'Blue Lake': 'Midwest', 'Bobcat Gas Storage': 'South Central', 'Cadeville Gas Storage': 'South Central', 'Caledonia Gas Storage': 'East',
        'Central Valley Gas Storage': 'Pacific', 'Clay Basin': 'Mountain', 'Clear Creek Storage': 'Mountain', 'Consumers Energy': 'Midwest',
        'DTE Gas Company': 'Midwest', 'Dominion Energy Questar Pipeline': 'Mountain', 'Dominion Energy Transmission': 'East',
        'Egan Hub Partners': 'South Central', 'Equitrans': 'East', 'Freebird Gas Storage': 'South Central', 'Gill Ranch Gas': 'Pacific',
        'Golden Triangle Storage': 'South Central', 'Gulf South Pipeline Company': 'South Central', 'Honeoye Storage Corporation': 'East',
        'Iroquois Gas/TransCanada': 'East', 'KINDER MORGAN LA PIPELINE': 'South Central', 'Kirby Hills Natural Gas Storage': 'Pacific',
        'Lodi Gas Services': 'Pacific', 'Midcontinent Express': 'South Central', 'Mississippi Hub': 'South Central',
        'Moss Bluff': 'South Central', 'NATIONAL FUEL GAS SUPPLY': 'East', 'NGPL-Midcon': 'South Central',
        'Natural Gas Pipeline Co of America': 'Midwest', 'Nicor Gas': 'Midwest', 'Northern Natural Gas Company': 'Midwest',
        'Northwest Pipeline Corporation': 'Mountain', 'Oneok Gas Storage': 'Midwest', 'Oneok Westex Transmission': 'South Central',
        'PG&E Gas Transmission': 'Pacific', 'Panhandle Eastern Pipe Line': 'Midwest', 'Peoples Gas Light & Coke Co': 'Midwest',
        'Perryville': 'South Central', 'Petal Gas Storage': 'South Central', 'Pine Prairie Energy Center': 'South Central',
        'Rockies Express Pipeline': 'Mountain', 'SENECA GAS STORAGE': 'East', 'Sonat': 'South Central',
        'Southern Natural Gas Company': 'South Central', 'Southern Pines Energy Center': 'South Central', 'Southwest Gas Corporation': 'Pacific',
        'Spire Storage': 'Mountain', 'Stagecoach Gas Services': 'East', 'Steuben Gas Storage': 'East',
        'Tennessee Gas Pipeline Co': 'East', 'Texas Gas Transmission': 'South Central', 'Transco': 'East',
        'Tres Palacios Gas Storage': 'South Central', 'Trunkline Gas Company': 'Midwest', 'WBI Energy Transmission': 'Mountain'
    }
    return storage_map

def process_storage_change_data(info_dir):
    print("Processing Criterion Storage Change Data...")
    storage_file = info_dir / 'CriterionStorageChange.csv'
    if not storage_file.exists(): return pd.DataFrame()
    storage_to_region_map = get_storage_to_region_map()
    if storage_to_region_map is None: return pd.DataFrame()
    df_storage_long = pd.read_csv(storage_file)
    df_storage_long.rename(columns={'eff_gas_day': 'Date', 'daily_storage_change': 'Value'}, inplace=True)
    df_storage_long['Date'] = pd.to_datetime(df_storage_long['Date'])
    df_storage_long['Region'] = df_storage_long['storage_name'].map(storage_to_region_map)
    df_storage_long.dropna(subset=['Region'], inplace=True)
    df_storage_long['Feature_Name'] = df_storage_long.apply(lambda row: f"{row['Region']}_Storage_{sanitize_name(row['storage_name'])}", axis=1)
    df_storage_granular = df_storage_long.pivot_table(index='Date', columns='Feature_Name', values='Value', aggfunc='mean')
    df_storage_regional = df_storage_long.pivot_table(index='Date', columns='Region', values='Value', aggfunc='sum')
    df_storage_regional.columns = [f'{sanitize_name(col)}_Criterion_Storage_Change' for col in df_storage_regional.columns]
    df_final = df_storage_regional.join(df_storage_granular, how='outer').fillna(0)
    print("  - Criterion Storage Change data processing complete.")
    return df_final

def process_eia_ground_truth(info_dir, full_date_index):
    print("Processing EIA Ground Truth Data (Target and Inventory)...")
    changes_file = info_dir / 'EIAchanges.csv'
    totals_file = info_dir / 'EIAtotals.csv'
    if not changes_file.exists() or not totals_file.exists(): return pd.DataFrame(), None
    df_changes = pd.read_csv(changes_file)
    df_changes['Date'] = pd.to_datetime(df_changes['Period'])
    df_changes = df_changes.set_index('Date')[['Lower 48 States Storage Change (Bcf)']].rename(columns={'Lower 48 States Storage Change (Bcf)': 'Target_Weekly_Storage_Change'})
    last_eia_date = df_changes.index.max()
    df_totals = pd.read_csv(totals_file)
    df_totals['Date'] = pd.to_datetime(df_totals['Period'])
    df_totals = df_totals.set_index('Date')
    inventory_columns = ['Lower 48 States Storage (Bcf)', 'East Region Storage (Bcf)', 'Midwest Region Storage (Bcf)', 'South Central Region Storage (Bcf)', 'Mountain Region Storage (Bcf)', 'Pacific Region Storage (Bcf)']
    df_totals_subset = df_totals[inventory_columns].copy()
    df_totals_subset.columns = ['Inv_' + sanitize_name(col) for col in df_totals_subset.columns]
    for col in df_totals_subset.columns:
        if col.startswith('Inv_'):
            df_totals_subset[f'end_of_prior_week_{col}'] = df_totals_subset[col].shift(1)
    df_totals_subset['prior_week_actual_change'] = df_changes['Target_Weekly_Storage_Change'].shift(1)

    df_region_changes = pd.DataFrame(index=df_totals_subset.index)
    region_map = {
        'East': 'Inv_East_Region_Storage_Bcf', 'Midwest': 'Inv_Midwest_Region_Storage_Bcf',
        'Mountain': 'Inv_Mountain_Region_Storage_Bcf', 'Pacific': 'Inv_Pacific_Region_Storage_Bcf',
        'SouthCentral': 'Inv_South_Central_Region_Storage_Bcf',
    }
    for region, col in region_map.items():
        if col in df_totals_subset.columns:
            df_region_changes[f'Target_{region}_Change'] = df_totals_subset[col].diff()

    df_weekly = df_changes.join(df_totals_subset, how='outer')
    df_weekly = df_weekly.join(df_region_changes, how='outer')
    df_daily_aligned = df_weekly.reindex(full_date_index, method='bfill')
    print("  - EIA Ground Truth processing complete.")
    return df_daily_aligned, last_eia_date

def create_intra_week_features(df):
    print("Engineering Intra-Week Features...")
    df_out = df.copy()
    eia_week_grouper = pd.Grouper(freq='W-FRI')
    if 'CONUS_Platts_DryGasProduction' in df_out.columns and 'CONUS_Platts_USDemand' in df_out.columns:
        df_out['daily_imbalance'] = df_out['CONUS_Platts_DryGasProduction'] - df_out['CONUS_Platts_USDemand']
        df_out['cumulative_imbalance'] = df_out.groupby(eia_week_grouper)['daily_imbalance'].cumsum()
    day_map = {5: 1, 6: 2, 0: 3, 1: 4, 2: 5, 3: 6, 4: 7}
    df_out['day_of_eia_week'] = df_out.index.dayofweek.map(day_map)
    df_out['days_until_report'] = 7 - df_out['day_of_eia_week']
    print("  - Intra-Week feature engineering complete.")
    return df_out

def add_calendar_and_lagged_features(df):
    """Engineers holiday, seasonal, and lagged features to prevent data leakage."""
    print("Engineering Calendar & Lagged Features...")
    df_out = df.copy()
    
    # Calendar Features
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df_out.index.min(), end=df_out.index.max())
    df_out['is_holiday'] = df_out.index.normalize().isin(holidays)
    df_out['is_holiday_week'] = df_out['is_holiday'].rolling(window='7D', min_periods=1).max().astype(int)
    df_out['is_injection_season'] = df_out.index.month.isin(range(4, 11)).astype(int)

    # Lagged Features
    # Identify key dynamic columns to lag
    cols_to_lag = [col for col in df_out.columns if 'CONUS_Platts' in col or '_Weather_' in col]
    for col in cols_to_lag:
        df_out[f'{col}_lag1'] = df_out[col].shift(1)

    print("  - Calendar & Lagged feature engineering complete.")
    return df_out

# --- Main Execution Pipeline ---

if __name__ == '__main__':
    print("--- Running Full Feature Engineering Pipeline ---")

    feature_processors = [
        process_weather_data,
        process_platts_conus_data,
        process_power_data,
        process_nuclear_data,
        process_lng_data,
        process_criterion_extra_data,
        process_fundy_data,
        process_storage_change_data,
    ]

    all_dfs = [func(INFO_DIR) for func in feature_processors]
    df_master = pd.concat(all_dfs, axis=1)
    df_master.sort_index(inplace=True)

    print("\n--- Joining Ground Truth and Engineering Final Features ---")
    df_eia_data, last_eia_date = process_eia_ground_truth(INFO_DIR, df_master.index)
    df_master = df_master.join(df_eia_data, how='left')

    df_master = create_intra_week_features(df_master)
    df_master = add_calendar_and_lagged_features(df_master)

    print("Cleaning and saving final dataset...")
    
    start_date = pd.to_datetime('2018-01-06')
    df_master = df_master.loc[start_date:].copy()
    
    df_master.ffill(inplace=True)
    df_master.bfill(inplace=True)
    
    remaining_nans = df_master.isnull().sum().sum()
    if remaining_nans == 0:
        print("Successfully cleaned all missing values.")
    else:
        print(f"WARNING: {remaining_nans} missing values still remain. Please review.")

    export_path = OUTPUT_DIR / 'model_ready_feature_set.csv'
    df_master.to_csv(export_path)

    print(f"\nFinal model-ready feature set shape: {df_master.shape}")
    print(f"Data runs from {df_master.index.min().date()} to {df_master.index.max().date()}")
    print(f"Successfully saved final feature set to: {export_path}")
    print("\n--- Feature Engineering Complete ---")
