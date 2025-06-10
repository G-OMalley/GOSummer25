import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- Configuration & Mappings ---
SCRIPT_DIR = Path(__file__).resolve().parent
INFO_DIR = SCRIPT_DIR.parent / 'INFO'
REGIONS = ['East', 'Midwest', 'Mountain', 'Pacific', 'South Central']

CITY_TO_REGION_MAP = {
    'Atlanta GA': 'East', 'Boston MA': 'East', 'Buffalo NY': 'East',
    'Washington National DC': 'East', 'John F. Kennedy NY': 'East',
    'Philadelphia PA': 'East', 'Pittsburgh PA': 'East',
    'Raleigh/Durham NC': 'East', 'Tampa FL': 'East',
    'Chicago IL': 'Midwest', 'Detroit MI': 'Midwest',
    'Denver CO': 'Mountain',
    'Los Angeles CA': 'Pacific', 'Seattle WA': 'Pacific', 'San Francisco CA': 'Pacific',
    'Houston TX': 'South Central', 'Little Rock AR': 'South Central',
    'New Orleans LA': 'South Central', 'Oklahoma City OK': 'South Central'
}
LNG_ITEM_TO_REGION_MAP = {
    'Calcasieu Pass LNG Feed Gas': 'South Central', 'Cameron LNG Feed Gas': 'South Central',
    'Corpus Christi LNG Feed Gas': 'South Central', 'Cove Point LNG Feed Gas': 'East',
    'Elba Island LNG Feed Gas': 'South Central', 'Freeport LNG Feed Gas': 'South Central',
    'Sabine Pass LNG Feed Gas': 'South Central', 'US LNG Exports - Plaquemines LNG': 'South Central'
}
CRITERION_EXTRA_MAP = {
    'CONUS - STORAGE': 'CONUS_Criterion_Storage', 'Total Demand - California': 'Pacific_Criterion_Demand_CA',
    'Total Demand - Lower 48': 'CONUS_Criterion_Total_Demand', 'Total Demand - Midwest': 'Midwest_Criterion_Total_Demand',
    'Total Demand - Northeast': 'East_Criterion_Total_Demand', 'Total Demand - Pacific Northwest': 'Pacific_Criterion_Demand_PNW',
    'Total Demand - Rockies': 'Mountain_Criterion_Total_Demand', 'Total Demand - Rockies - SW': 'Mountain_Criterion_Demand_SW',
    'Total Demand - Rockies - Upper': 'Mountain_Criterion_Demand_Upper', 'Total Demand - South Central': 'South_Central_Criterion_Total_Demand',
    'Total Demand - Southeast - Florida': 'East_Criterion_Demand_SE_FL', 'Total Demand - Southeast - Other': 'East_Criterion_Demand_SE_Other'
}
STORAGE_TO_REGION_MAP = {
    'ANR Pipeline Company': 'South Central', 'ANR Storage Company': 'Midwest', 'Arcadia Gas Storage': 'South Central',
    'Avon Storage': 'Midwest', 'Bay Gas Storage Company': 'South Central', 'Bear Creek Storage Company': 'South Central',
    'Beckman': 'Midwest', 'Bethel': 'South Central', 'Bistineau': 'South Central', 'Black-Sulphur': 'South Central',
    'Blue Lake': 'Midwest', 'Bobcat Gas Storage': 'South Central', 'Boling': 'South Central', 'Bonanza': 'Mountain',
    'Cadeville Gas Storage': 'South Central', 'Caledonia Gas Storage': 'East', 'Cameron': 'South Central',
    'Carlton Gas Storage': 'Midwest', 'Carson': 'South Central', 'Cashion': 'South Central', 'Cayuta': 'East',
    'Central New York Oil and Gas': 'East', 'Central Valley Gas Storage': 'Pacific', 'Ceres': 'Midwest',
    'Clay Basin': 'Mountain', 'Clear Creek Storage': 'Mountain', 'Columbus Gas Transmission': 'East',
    'Como': 'Midwest', 'Consolidated Gas Supply': 'East', 'Consumers Energy': 'Midwest', 'Copiah': 'South Central',
    'Cunningham': 'Midwest', 'DTE Gas Company': 'Midwest', 'DeSoto': 'South Central',
    'Dominion Energy Questar Pipeline': 'Mountain', 'Dominion Energy Transmission': 'East', 'Duck Lake': 'South Central',
    'East Ohio Gas Company': 'East', 'Egan Hub Partners': 'South Central', 'Equitrans': 'East',
    'Fair-p-t': 'South Central', 'Fidelity': 'Midwest', 'Freebird Gas Storage': 'South Central', 'GTI-Dawn': 'Midwest',
    'Gas T-D': 'East', 'Gibson': 'South Central', 'Gill Ranch Gas': 'Pacific', 'Glenarm': 'Midwest',
    'Golden Triangle Storage': 'South Central', 'Gram-p-t': 'South Central', 'Green River': 'Mountain',
    'Greenbrier': 'East', 'Grom-p-t': 'South Central', 'Guardian': 'Midwest', 'Gulf South Pipeline Company': 'South Central',
    'Gulf-p-t': 'South Central', 'HIOS': 'South Central', 'Hattiesburg': 'South Central', 'Hav-p-t': 'South Central',
    'Henry': 'South Central', 'Hillman': 'Midwest', 'Honeoye Storage Corporation': 'East',
    'Iroquois Gas/TransCanada': 'East', 'Jackson': 'South Central', 'James-p-t': 'South Central', 'Jena': 'South Central',
    'K-p-t': 'South Central', 'Kankakee': 'Midwest', 'Kansas-Nebraska Big Well': 'Midwest', 'Katy': 'South Central',
    'KINDER MORGAN LA PIPELINE': 'South Central', 'Kirby Hills Natural Gas Storage': 'Pacific', 'L-p-t': 'South Central',
    'Lakeside': 'Midwest', 'Lauber': 'Midwest', 'Lea-p-t': 'South Central', 'Leroy': 'Mountain', 'Lincoln': 'Midwest',
    'Lodi Gas Services': 'Pacific', 'Loudon': 'Midwest', 'Lov-p-t': 'South Central', 'Lowry': 'Mountain',
    'M-p-t': 'South Central', 'MGS': 'South Central', 'Manc-p-t': 'South Central', 'Manlove': 'Midwest',
    'Markham': 'South Central', 'Meeker': 'Mountain', 'Michigan Gas Utilities': 'Midwest', 'Mid-p-t': 'South Central',
    'Midcontinent Express': 'South Central', 'Midla': 'Midwest', 'Mississippi Hub': 'South Central',
    'Missouri Gas Energy': 'Midwest', 'Moss Bluff': 'South Central', 'Mul-p-t': 'South Central', 'N-p-t': 'South Central',
    'NATIONAL FUEL GAS SUPPLY': 'East', 'NGPL-Midcon': 'South Central', 'NGPL-Texoma': 'South Central',
    'NY-p-t': 'East', 'Napoleonville': 'South Central', 'Natural Gas Pipeline Co of America': 'Midwest',
    'Nautilus': 'South Central', 'Nelson': 'Midwest', 'Nicor Gas': 'Midwest', 'North Lansing': 'East',
    'Northern Illinois Gas Company': 'Midwest', 'Northern Indiana Public Service Co': 'Midwest',
    'Northern Natural Gas Company': 'Midwest', 'Northwest Pipeline Corporation': 'Mountain', 'O-p-t': 'South Central',
    'Oak Grove': 'South Central', 'Oakford': 'East', 'Oneok Gas Storage': 'Midwest',
    'Oneok Westex Transmission': 'South Central', 'PG&E Gas Transmission': 'Pacific',
    'Panhandle Eastern Pipe Line': 'Midwest', 'Peoples Gas Light & Coke Co': 'Midwest', 'Perryville': 'South Central',
    'Petal Gas Storage': 'South Central', 'Piceance': 'Mountain', 'Pine Prairie Energy Center': 'South Central',
    'Piv-p-t': 'South Central', 'Port Barre Investments': 'South Central', 'Portland Natural Gas': 'East',
    'Putnam': 'Midwest', 'Quest': 'Mountain', 'R-p-t': 'South Central', 'Red-p-t': 'South Central',
    'Rendezvous': 'Mountain', 'Riner': 'Mountain', 'Rockies Express Pipeline': 'Mountain', 'Ryckman': 'Mountain',
    'S-p-t': 'South Central', 'SEMCO Energy Gas Company': 'Midwest', 'SENECA GAS STORAGE': 'East',
    'Sab-p-t': 'South Central', 'Sayre': 'Midwest', 'Sharon': 'South Central', 'Sher-p-t': 'South Central',
    'Silo': 'South Central', 'So-p-t': 'South Central', 'Sonat': 'South Central', 'Southern Natural Gas Company': 'South Central',
    'Southern Pines Energy Center': 'South Central', 'Southwest Gas Corporation': 'Pacific', 'Spearman': 'South Central',
    'Spire Storage': 'Mountain', 'St. Clair': 'Midwest', 'Stagecoach Gas Services': 'East', 'Starks': 'South Central',
    'Steuben Gas Storage': 'East', 'Sulphur': 'South Central', 'T-p-t': 'South Central', 'Tall-p-t': 'South Central',
    'Tennessee Gas Pipeline Co': 'East', 'Terre-p-t': 'South Central', 'Tex-p-t': 'South Central',
    'Texas Gas Transmission': 'South Central', 'Texasok': 'South Central', 'Trans-p-t': 'East', 'Transco': 'East',
    'Tres Palacios Gas Storage': 'South Central', 'Triumph': 'East', 'Tropic': 'South Central',
    'Trunkline Gas Company': 'Midwest', 'U-p-t': 'South Central', 'V-p-t': 'South Central', 'Vector Pipeline': 'Midwest',
    'W-p-t': 'South Central', 'WBI Energy Transmission': 'Mountain', 'WE Energies': 'Midwest', 'Waha': 'South Central',
    'Washington': 'Midwest', 'Waverly': 'Midwest', 'Wel-p-t': 'South Central', 'Western Gas Interstate': 'Mountain',
    'Wheeler Gas Storage': 'East', 'White-p-t': 'South Central', 'Wilson': 'East', 'Woolfolk': 'Midwest'
}
POWER_ITEM_TO_REGION_MAP = {
    'AESO': ['Mountain'],
    'BPA': ['Pacific'],
    'CAISO': ['Pacific'],
    'ERCOT': ['South Central'],
    'IESO': ['Midwest', 'East'],
    'ISONE': ['East'],
    'MISO': ['Midwest', 'South Central'],
    'NYISO': ['East'],
    'PJM': ['East'],
    'SPP': ['Midwest', 'South Central']
}


# --- Feature Engineering Functions ---

def create_time_features(df):
    """Creates time-series features from the datetime index."""
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    return df

def process_weather_data(info_dir_path):
    """
    Loads weather data, creates regional aggregates AND granular city-level features
    with region names prepended for clarity and easier filtering.
    """
    print("Processing Weather Data...")
    weather_file = info_dir_path / 'WEATHER.csv'
    if not weather_file.exists(): return pd.DataFrame()
    df_weather = pd.read_csv(weather_file)
    df_weather['Date'] = pd.to_datetime(df_weather['Date'])
    
    # Map regions to each city
    df_weather['Region'] = df_weather['City Title'].map(CITY_TO_REGION_MAP)
    df_weather.dropna(subset=['Region'], inplace=True)

    # --- Regional Aggregates (e.g., East_HDD) ---
    agg_dict = {'CDD':'sum', 'HDD':'sum'}
    df_regional_agg = df_weather.groupby(['Date', 'Region']).agg(agg_dict).reset_index()
    df_regional_pivot = df_regional_agg.pivot(index='Date', columns='Region', values=list(agg_dict.keys()))
    df_regional_pivot.columns = [f'{region}_{stat.replace(" ", "_")}' for stat, region in df_regional_pivot.columns]
    
    # --- Granular City-Level Features (e.g., Midwest_City_Detroit_MI_Avg_Temp) ---
    # FIX: Prepend the region to each granular feature name
    df_city_pivots = []
    for region, df_group in df_weather.groupby('Region'):
        df_pivot = df_group.pivot(index='Date', columns='City Title', values=['Avg Temp', 'CDD', 'HDD'])
        df_pivot.columns = [f"{region}_City_{city.replace(' ', '_')}_{stat}" for stat, city in df_pivot.columns]
        df_city_pivots.append(df_pivot)
    df_city_granular = pd.concat(df_city_pivots, axis=1)

    # --- Combine aggregate and granular features ---
    df_final = df_regional_pivot.join(df_city_granular, how='outer')
    print("Weather data processing complete.")
    return df_final

def process_fundy_data(info_dir_path):
    print("\nProcessing Fundamental Data (Fundy)...")
    fundy_file = info_dir_path / 'Fundy.csv'
    if not fundy_file.exists(): return pd.DataFrame()
    df_fundy_long = pd.read_csv(fundy_file)
    df_fundy_long['Date'] = pd.to_datetime(df_fundy_long['Date'])
    df_fundy_wide = df_fundy_long.pivot_table(index='Date', columns='item', values='value', aggfunc='mean')
    df_fundy_wide.columns = [f"Fundy_{col.strip().replace(' ', '_').replace('-', '_')}" for col in df_fundy_wide.columns]
    print("Fundamental data processing complete.")
    return df_fundy_wide

def process_lng_data(info_dir_path):
    print("\nProcessing LNG Data...")
    lng_file = info_dir_path / 'CriterionLNGHist.csv'
    if not lng_file.exists(): return pd.DataFrame()
    df_lng_long = pd.read_csv(lng_file)
    df_lng_long['Date'] = pd.to_datetime(df_lng_long['Date'])
    
    # FIX: Prepend region to granular features
    df_lng_long['Region'] = df_lng_long['Item'].map(LNG_ITEM_TO_REGION_MAP)
    df_lng_long.dropna(subset=['Region'], inplace=True)
    
    # Granular (e.g., South_Central_LNG_Cameron_LNG_Feed_Gas)
    df_lng_long['Feature_Name'] = df_lng_long.apply(
        lambda row: f"{row['Region']}_LNG_{row['Item'].replace(' ', '_').replace('-', '_')}", axis=1)
    df_lng_granular = df_lng_long.pivot_table(index='Date', columns='Feature_Name', values='Value', aggfunc='mean')

    # Regional (e.g., South_Central_LNG_Feedgas_Total)
    df_regional_lng = df_lng_long.groupby(['Date', 'Region'])['Value'].sum().reset_index()
    df_lng_regional = df_regional_lng.pivot(index='Date', columns='Region', values='Value')
    df_lng_regional.columns = [f'{col.replace(" ", "_")}_LNG_Feedgas_Total' for col in df_lng_regional.columns]
    
    df_final = df_lng_regional.join(df_lng_granular, how='outer').fillna(0)
    print("LNG data processing complete.")
    return df_final

def process_criterion_extra_data(info_dir_path):
    print("\nProcessing Criterion Extra Data...")
    extra_file = info_dir_path / 'CriterionExtra.csv'
    if not extra_file.exists(): return pd.DataFrame()
    df_extra_long = pd.read_csv(extra_file)
    df_extra_long['Date'] = pd.to_datetime(df_extra_long['Date'])
    df_filtered = df_extra_long[df_extra_long['item'].isin(CRITERION_EXTRA_MAP.keys())].copy()
    if df_filtered.empty: return pd.DataFrame()
    df_filtered['Feature_Name'] = df_filtered['item'].map(CRITERION_EXTRA_MAP)
    df_extra_wide = df_filtered.pivot_table(index='Date', columns='Feature_Name', values='value', aggfunc='mean')
    print("Criterion Extra data processing complete.")
    return df_extra_wide

def process_criterion_storage_change(info_dir_path):
    print("\nProcessing Criterion Storage Change Data...")
    storage_file = info_dir_path / 'CriterionStorageChange.csv'
    if not storage_file.exists(): return pd.DataFrame()
    df_storage_long = pd.read_csv(storage_file)
    
    # FIX: Prepend region to granular features
    df_storage_long['Region'] = df_storage_long['storage_name'].map(STORAGE_TO_REGION_MAP)
    df_storage_long.dropna(subset=['Region'], inplace=True)
    
    # Granular (e.g., Midwest_Storage_ANR_Storage_Company)
    df_storage_long['Feature_Name'] = df_storage_long.apply(
        lambda row: f"{row['Region']}_Storage_{row['storage_name'].replace(' ', '_').replace('-', '_')}", axis=1)
    df_storage_granular = df_storage_long.pivot_table(index='eff_gas_day', columns='Feature_Name', values='daily_storage_change', aggfunc='mean')
    df_storage_granular.index = pd.to_datetime(df_storage_granular.index)

    # Regional (e.g., Midwest_Criterion_Storage_Change)
    df_storage_long.rename(columns={'eff_gas_day': 'Date', 'daily_storage_change': 'Value'}, inplace=True)
    df_storage_long['Date'] = pd.to_datetime(df_storage_long['Date'])
    df_regional_storage = df_storage_long.groupby(['Date', 'Region'])['Value'].sum().reset_index()
    df_storage_regional = df_regional_storage.pivot(index='Date', columns='Region', values='Value')
    df_storage_regional.columns = [f'{col.replace(" ", "_")}_Criterion_Storage_Change' for col in df_storage_regional.columns]
    
    df_final = df_storage_regional.join(df_storage_granular, how='outer').fillna(0)
    print("Criterion Storage Change data processing complete.")
    return df_final

def process_power_data(info_dir_path):
    """
    Loads and processes Platts Power Fundamentals data, mapping items to regions.
    """
    print("\nProcessing Platts Power Data...")
    power_file = info_dir_path / 'PlattsPowerFundy.csv'
    if not power_file.exists():
        print("PlattsPowerFundy.csv not found. Skipping.")
        return pd.DataFrame()

    df_power_long = pd.read_csv(power_file)

    if 'Value' not in df_power_long.columns:
        print("ERROR: 'Value' column not found in PlattsPowerFundy.csv. Skipping.")
        return pd.DataFrame()

    df_power_long['Date'] = pd.to_datetime(df_power_long['Date'])
    df_power_long.dropna(subset=['Item', 'Value'], inplace=True)

    def get_regions(item_name):
        for prefix, regions in POWER_ITEM_TO_REGION_MAP.items():
            if item_name.strip().startswith(prefix):
                return regions
        return None

    df_power_long['Region'] = df_power_long['Item'].apply(get_regions)
    df_power_long.dropna(subset=['Region'], inplace=True)

    df_exploded = df_power_long.explode('Region')

    df_exploded['Feature_Name'] = df_exploded.apply(
        lambda row: f"{row['Region']}_Power_{row['Item'].strip().replace(' ', '_').replace('-', '_')}",
        axis=1
    )

    df_power_wide = df_exploded.pivot_table(
        index='Date',
        columns='Feature_Name',
        values='Value',
        aggfunc='mean'
    )
    
    print("Platts Power data processing complete.")
    return df_power_wide


def process_target_and_inventory(info_dir_path):
    print("\nProcessing Target and Inventory Data...")
    target_file = info_dir_path / 'PlattsCONUSFundamentalsHIST.csv'
    inventory_file = info_dir_path / 'EIAtotals.csv'
    if not target_file.exists() or not inventory_file.exists():
        print("CRITICAL ERROR: Target or Inventory file not found.")
        return pd.DataFrame()
        
    df_target = pd.read_csv(target_file)[['GasDate', 'ImpliedStorageChange']]
    df_target.rename(columns={'GasDate': 'Date', 'ImpliedStorageChange': 'Daily_Storage_Change'}, inplace=True)
    df_target['Date'] = pd.to_datetime(df_target['Date'])
    df_target.set_index('Date', inplace=True)
    
    df_inventory = pd.read_csv(inventory_file)
    df_inventory.rename(columns={'Period': 'Date'}, inplace=True)
    df_inventory['Date'] = pd.to_datetime(df_inventory['Date'])
    df_inventory.set_index('Date', inplace=True)
    df_inventory_daily = df_inventory.resample('D').ffill()
    df_inventory_daily.columns = ['Inv_' + col.replace(' ', '_').replace('(Bcf)', '').strip() for col in df_inventory_daily.columns]
    
    df_final = df_target.join(df_inventory_daily, how='left')
    print("Target and Inventory data processing complete.")
    return df_final

def handle_missing_data(df):
    """
    Handles missing data by dropping columns with too many NaNs and interpolating the rest.
    """
    print("\n--- Cleaning Missing Data ---")
    
    percent_missing = df.isnull().sum() / len(df) * 100
    cols_to_drop = percent_missing[percent_missing > 2].index.tolist()
        
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"Dropped {len(cols_to_drop)} columns with >2% missing data:")
        for col in cols_to_drop:
            print(f"  - {col}")
    else:
        print("No columns had more than 2% missing data.")

    print("\nInterpolating remaining missing values...")
    df.interpolate(method='linear', limit_direction='both', inplace=True)

    remaining_nans = df.isnull().sum().sum()
    if remaining_nans == 0:
        print("Successfully cleaned all missing values from feature columns.")
    else:
        print(f"WARNING: {remaining_nans} missing values still remain. Please review.")
        
    return df


# --- Main execution block ---
if __name__ == '__main__':
    print("--- Running Full Feature Engineering Pipeline ---")
    
    feature_processors = [
        process_weather_data,
        process_fundy_data,
        process_lng_data,
        process_criterion_extra_data,
        process_criterion_storage_change,
        process_power_data,
        process_target_and_inventory
    ]
    
    df_master = pd.DataFrame()
    for func in feature_processors:
        df_new = func(INFO_DIR)
        if df_new is not None and not df_new.empty:
            df_master = df_master.join(df_new, how='outer') if not df_master.empty else df_new
            
    df_master = create_time_features(df_master)
    df_master.sort_index(inplace=True)
    
    print("\n--- Trimming Data and Analyzing Gaps ---")
    start_date = pd.to_datetime('2018-01-06')
    eia_changes_file = INFO_DIR / 'EIAchanges.csv'
    if not eia_changes_file.exists():
        raise FileNotFoundError(f"Required file for trimming not found: {eia_changes_file}")
    df_eia_dates = pd.read_csv(eia_changes_file)
    df_eia_dates['Period'] = pd.to_datetime(df_eia_dates['Period'])
    last_eia_date = df_eia_dates['Period'].max()
    cut_off_date = last_eia_date + pd.Timedelta(days=7)

    print(f"Original data runs from {df_master.index.min().date()} to {df_master.index.max().date()}.")
    print(f"Trimming data from {start_date.date()} to {cut_off_date.date()}.")
    
    df_trimmed = df_master.loc[start_date:cut_off_date].copy()
    print(f"Data trimmed. Shape after trimming: {df_trimmed.shape}")
    
    df_clean = handle_missing_data(df_trimmed)

    OUTPUT_DIR = SCRIPT_DIR / 'output'
    OUTPUT_DIR.mkdir(exist_ok=True)
    export_path = OUTPUT_DIR / 'final_feature_set.csv'
    df_clean.to_csv(export_path)
    print(f"\nFinal feature set saved to: {export_path}")
    
    print("\n--- Feature Engineering Complete ---")