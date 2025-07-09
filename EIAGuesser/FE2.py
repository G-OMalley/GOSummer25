import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Suppress warnings for cleaner output, e.g., from date parsing
warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# --- 1. CONFIGURATION & MAPPINGS ---
# =============================================================================
print("--- Initializing Feature Engineering Pipeline ---")

# --- File Paths ---
try:
    # Assumes the script is in a subfolder of the main project directory
    SCRIPT_DIR = Path(__file__).resolve().parent
    INFO_DIR = SCRIPT_DIR.parent / 'INFO'
except NameError:
    # Fallback for interactive environments (like Jupyter)
    INFO_DIR = Path.cwd() / 'INFO'
    print(f"Warning: Could not determine script directory. Assuming INFO is at: {INFO_DIR}")

# --- Core Mappings ---
# Helper to standardize dictionary keys for robust matching
def clean_dict_keys(d):
    return {str(k).strip().lower(): v for k, v in d.items()}

CITY_SYMBOL_TO_REGION_MAP = clean_dict_keys({
    'KATL': 'East', 'KBOS': 'East', 'KBUF': 'East', 'KDCA': 'East', 'KJFK': 'East',
    'KPHL': 'East', 'KPIT': 'East', 'KRDU': 'East', 'KTPA': 'East', 'KORD': 'Midwest',
    'KDTW': 'Midwest', 'KMSP': 'Midwest', 'KDEN': 'Mountain', 'KLAX': 'Pacific',
    'KSEA': 'Pacific', 'KSFO': 'Pacific', 'KIAH': 'South Central', 'KLIT': 'South Central',
    'KMSY': 'South Central', 'KOKC': 'South Central'
})

ISO_TO_REGION_MAP = clean_dict_keys({
    'AESO': ['Mountain'], 'BPA': ['Pacific'], 'CAISO': ['Pacific'],
    'ERCOT': ['South Central'], 'IESO': ['Midwest', 'East'],
    'ISONE': ['East'], 'MISO': ['Midwest', 'South Central'],
    'NYISO': ['East'], 'PJM': ['East'], 'SPP': ['Midwest', 'South Central']
})

STATE_TO_REGION_MAP = {
    'CT': 'East', 'DE': 'East', 'FL': 'East', 'GA': 'East', 'MA': 'East', 'MD': 'East', 'ME': 'East', 'NC': 'East',
    'NH': 'East', 'NJ': 'East', 'NY': 'East', 'OH': 'East', 'PA': 'East', 'RI': 'East', 'SC': 'East', 'VA': 'East',
    'VT': 'East', 'WV': 'East', 'IL': 'Midwest', 'IN': 'Midwest', 'IA': 'Midwest', 'KY': 'Midwest', 'MI': 'Midwest',
    'MN': 'Midwest', 'MO': 'Midwest', 'TN': 'Midwest', 'WI': 'Midwest', 'AZ': 'Mountain', 'CO': 'Mountain',
    'ID': 'Mountain', 'MT': 'Mountain', 'NE': 'Mountain', 'NV': 'Mountain', 'NM': 'Mountain', 'ND': 'Mountain',
    'SD': 'Mountain', 'UT': 'Mountain', 'WY': 'Mountain', 'CA': 'Pacific', 'OR': 'Pacific', 'WA': 'Pacific',
    'AL': 'South Central', 'AR': 'South Central', 'KS': 'South Central', 'LA': 'South Central', 'MS': 'South Central',
    'OK': 'South Central', 'TX': 'South Central'
}

# =============================================================================
# --- 2. DATA LOADING & HELPER FUNCTIONS ---
# =============================================================================
def _safe_read_csv(file_path, **kwargs):
    """A helper function to safely read CSV files, reporting errors."""
    if not file_path.exists():
        print(f"  - File not found: {file_path.name}. Skipping.")
        return pd.DataFrame()
    try:
        print(f"  - Reading {file_path.name}...")
        return pd.read_csv(file_path, **kwargs)
    except Exception as e:
        print(f"  - Error reading {file_path.name}: {e}. Skipping.")
        return pd.DataFrame()

def load_all_data(info_dir):
    """Loads all necessary daily data files from the INFO directory."""
    print("\n--- Loading All Daily Data Sources ---")
    data = {
        "eia_changes": _safe_read_csv(info_dir / 'EIAchanges.csv'),
        "fundy": _safe_read_csv(info_dir / 'Fundy.csv'),
        "weather": _safe_read_csv(info_dir / 'WEATHER.csv'),
        "power_fundy": _safe_read_csv(info_dir / 'PlattsPowerFundy.csv'),
        "eia_totals": _safe_read_csv(info_dir / 'EIAtotals.csv'),
        "storage_change": _safe_read_csv(info_dir / 'CriterionStorageChange.csv'),
        "locs_list": _safe_read_csv(info_dir / 'locs_list.csv')
    }
    date_cols = {
        "eia_changes": "Period", "fundy": "Date", "weather": "Date",
        "power_fundy": "Date", "eia_totals": "Period", "storage_change": "eff_gas_day"
    }
    for name, df in data.items():
        if name in date_cols and not df.empty and date_cols[name] in df.columns:
            df[date_cols[name]] = pd.to_datetime(df[date_cols[name]], errors='coerce')
            df.dropna(subset=[date_cols[name]], inplace=True)
            print(f"    - Processed dates for {name}.")
    return data

def calculate_slope(series):
    """Calculates the slope of a time series using linear regression."""
    series = series.dropna()
    if len(series) < 2:
        return 0
    x = np.arange(len(series))
    slope, _ = np.polyfit(x, series, 1)
    return slope

# =============================================================================
# --- 3. DAILY FEATURE ENGINEERING MODULE ---
# =============================================================================
def process_storage_data(df_storage_change, df_locs):
    """Processes granular storage change data and maps it to regions."""
    print("  - Processing Granular Storage data...")
    if df_storage_change.empty or df_locs.empty:
        print("    - Missing CriterionStorageChange.csv or locs_list.csv. Skipping.")
        return pd.DataFrame()

    df_locs_map = df_locs[['storage_name', 'state_name']].dropna().drop_duplicates()
    df_locs_map['Region'] = df_locs_map['state_name'].map(STATE_TO_REGION_MAP)
    
    df_merged = pd.merge(df_storage_change, df_locs_map, on='storage_name', how='left')
    df_merged.dropna(subset=['Region'], inplace=True)
    df_merged.rename(columns={'eff_gas_day': 'Date'}, inplace=True)

    df_regional = df_merged.groupby(['Date', 'Region'])['daily_storage_change'].sum().reset_index()
    df_regional.rename(columns={'daily_storage_change': 'Storage_Criterion_Regional_Sum'}, inplace=True)
    
    print("    - Created regional sum of daily storage changes.")

    df_regional.sort_values(by=['Region', 'Date'], inplace=True)
    df_regional['Storage_30d_Rolling_Avg'] = df_regional.groupby('Region')['Storage_Criterion_Regional_Sum'].transform(
        lambda x: x.rolling(window=30, min_periods=14).mean()
    )
    df_regional['Storage_Change_vs_30d_Rolling_Avg'] = df_regional['Storage_Criterion_Regional_Sum'] - df_regional['Storage_30d_Rolling_Avg']
    df_regional.drop(columns=['Storage_30d_Rolling_Avg'], inplace=True)
    print("    - Created storage flow anomaly vs. 30-day average.")
    
    return df_regional

def engineer_daily_features(data_dict):
    """Creates a unified DataFrame of daily features from all loaded sources."""
    print("\n--- Engineering Daily Features ---")
    all_daily_features = []

    # --- Weather ---
    if not data_dict["weather"].empty:
        print("  - Processing Weather data...")
        df_weather = data_dict["weather"].copy()
        df_weather['Region'] = df_weather['City Symbol'].str.strip().str.lower().map(CITY_SYMBOL_TO_REGION_MAP)
        df_weather.dropna(subset=['Region'], inplace=True)
        for metric in ['Min Temp', 'Max Temp', 'Avg Temp', 'HDD', 'CDD']:
            norm_col = f'10yr {metric}'
            if norm_col in df_weather.columns:
                df_weather[f'Weather_{metric}_vs_10yrNorm'] = df_weather[metric] - df_weather[norm_col]
        weather_cols_to_agg = ['HDD', 'CDD', 'Avg Temp'] + [col for col in df_weather.columns if '_vs_10yrNorm' in col]
        df_weather_regional = df_weather.groupby(['Date', 'Region'])[weather_cols_to_agg].sum().reset_index()
        all_daily_features.append(df_weather_regional)

    # --- Power Gen ---
    if not data_dict["power_fundy"].empty:
        print("  - Processing Power Generation data...")
        df_power = data_dict["power_fundy"].copy()
        df_power['ISO'] = df_power['Item'].str.split(' ').str[0]
        df_power['Region'] = df_power['ISO'].str.strip().str.lower().map(ISO_TO_REGION_MAP)
        df_power = df_power.explode('Region').dropna(subset=['Region'])
        df_gas_gen = df_power[df_power['Item'].str.contains("Natural Gas", na=False)]
        df_total_load = df_power[df_power['Item'].str.contains("Total Generation", na=False)]
        if not df_gas_gen.empty and not df_total_load.empty:
            df_gas_regional = df_gas_gen.groupby(['Date', 'Region'])['Value'].sum().reset_index().rename(columns={'Value': 'GasGen'})
            df_load_regional = df_total_load.groupby(['Date', 'Region'])['Value'].sum().reset_index().rename(columns={'Value': 'TotalLoad'})
            df_power_merged = pd.merge(df_gas_regional, df_load_regional, on=['Date', 'Region'], how='inner')
            df_power_merged['Power_GasShare'] = (df_power_merged['GasGen'] / df_power_merged['TotalLoad']).fillna(0)
            all_daily_features.append(df_power_merged[['Date', 'Region', 'Power_GasShare', 'TotalLoad']])

    # --- Fundy ---
    if not data_dict["fundy"].empty:
        print("  - Processing Fundy fundamentals...")
        df_fundy = data_dict["fundy"].copy()
        df_fundy[['Region', 'Metric']] = df_fundy['item'].str.split(' - ', expand=True, n=1)
        df_fundy.dropna(subset=['Region', 'Metric'], inplace=True)
        df_fundy['Metric'] = "Fundy_" + df_fundy['Metric'].str.replace(' ', '_')
        df_fundy_regional = df_fundy.pivot_table(index=['Date', 'Region'], columns='Metric', values='value').reset_index().fillna(0)
        all_daily_features.append(df_fundy_regional)

    # --- EIA Inventory ---
    if not data_dict["eia_totals"].empty:
        print("  - Processing EIA Inventory data...")
        df_totals = data_dict["eia_totals"].copy()
        df_totals.set_index('Period', inplace=True)
        df_totals_daily = df_totals.resample('D').ffill()
        for region in ['East', 'Midwest', 'Mountain', 'Pacific', 'South Central', 'Lower 48 States']:
            current_col, avg_col = f'{region} Region Storage (Bcf)', f'{region} Region 5-Year Average (Bcf)'
            if current_col in df_totals_daily.columns and avg_col in df_totals_daily.columns:
                df_totals_daily[f'Inventory_{region.replace(" ", "")}_vs_5yrAvg'] = df_totals_daily[current_col] - df_totals_daily[avg_col]
        id_vars = [col for col in df_totals_daily if '_vs_5yrAvg' in col]
        df_inv_melted = df_totals_daily.reset_index().melt(id_vars='Period', value_vars=id_vars, var_name='Metric', value_name='Inventory_vs_5yrAvg')
        df_inv_melted['Region'] = df_inv_melted['Metric'].str.split('_').str[1]
        df_inv_melted.rename(columns={'Period': 'Date'}, inplace=True)
        all_daily_features.append(df_inv_melted[['Date', 'Region', 'Inventory_vs_5yrAvg']])

    # --- Granular Storage ---
    df_storage = process_storage_data(data_dict["storage_change"], data_dict["locs_list"])
    if not df_storage.empty:
        all_daily_features.append(df_storage)

    # --- Combine ---
    if not all_daily_features:
        raise ValueError("No daily features were created. Check input files.")
    print("  - Combining all daily feature sources...")
    from functools import reduce
    df_daily_master = reduce(lambda left, right: pd.merge(left, right, on=['Date', 'Region'], how='outer'), all_daily_features)
    df_daily_master.sort_values(by=['Region', 'Date'], inplace=True)
    df_daily_master.set_index(['Date', 'Region'], inplace=True)
    df_daily_master = df_daily_master.groupby(level='Region').ffill(limit=4).bfill(limit=4)
    df_daily_master.reset_index(inplace=True)
    print(f"    - Master daily feature table created with shape: {df_daily_master.shape}")
    return df_daily_master

# =============================================================================
# --- 4. WEEKLY AGGREGATION & FEATURE ENRICHMENT ---
# =============================================================================
def aggregate_to_weekly(df_daily, df_eia_changes):
    """Aggregates daily features to the EIA weekly level and adds intra-week features."""
    print("\n--- Aggregating Daily Data to Weekly Features ---")
    if df_daily.empty or df_eia_changes.empty: return pd.DataFrame()

    print("  - Building precise Sat-Fri weekly mapper from EIA dates...")
    eia_fridays = sorted(pd.to_datetime(df_eia_changes['Period']).unique())
    weekly_ranges = [pd.DataFrame({'Date': pd.date_range(start=fri - pd.Timedelta(days=6), end=fri, freq='D'), 'Week_Ending_Friday': fri}) for fri in eia_fridays]
    if not weekly_ranges: raise ValueError("No EIA weeks could be constructed.")
    weekly_map = pd.concat(weekly_ranges, ignore_index=True)
    
    df_daily_with_weeks = pd.merge(df_daily, weekly_map, on='Date', how='left')
    df_daily_with_weeks.dropna(subset=['Week_Ending_Friday'], inplace=True)

    # --- A. Intra-Week Feature Engineering (NEW) ---
    print("  - Engineering intra-week features (slopes, std devs, flags)...")
    intra_week_features = []
    grouped = df_daily_with_weeks.groupby(['Week_Ending_Friday', 'Region'])
    
    # FIX: Check for column existence on the DataFrame, not the GroupBy object
    # Weather
    if 'HDD' in df_daily_with_weeks.columns:
        intra_week_features.append(grouped['HDD'].std().rename('HDD_std_7d'))
    if 'CDD' in df_daily_with_weeks.columns:
        intra_week_features.append(grouped['CDD'].max().rename('CDD_max_7d'))
    if 'Avg Temp' in df_daily_with_weeks.columns:
        intra_week_features.append(grouped['Avg Temp'].apply(lambda x: calculate_slope(x.tail(3))).rename('AvgTemp_slope_last3d'))

    # Fundamentals
    if 'Fundy_Power' in df_daily_with_weeks.columns:
        intra_week_features.append(grouped['Fundy_Power'].max().rename('Fundy_Power_max_7d'))
    if 'Fundy_LNGexp' in df_daily_with_weeks.columns:
        intra_week_features.append((grouped['Fundy_LNGexp'].max() > grouped['Fundy_LNGexp'].mean() * 1.5).rename('Fundy_LNGexp_surge_flag').astype(int))
    if 'Fundy_MexExp' in df_daily_with_weeks.columns:
        intra_week_features.append(grouped['Fundy_MexExp'].apply(lambda x: x.tail(1).iloc[0] - x.head(1).iloc[0] if len(x) > 1 else 0).rename('Fundy_MexExp_delta_week'))
    
    # Storage
    if 'Storage_Criterion_Regional_Sum' in df_daily_with_weeks.columns:
        intra_week_features.append(grouped['Storage_Criterion_Regional_Sum'].max().rename('Storage_Criterion_max_injection_7d'))
        intra_week_features.append(grouped['Storage_Criterion_Regional_Sum'].apply(lambda x: x.tail(5).std()).rename('Storage_Criterion_std_5d'))
        intra_week_features.append(grouped['Storage_Criterion_Regional_Sum'].last().rename('Storage_Criterion_last_day'))

    # Power
    if 'Power_GasShare' in df_daily_with_weeks.columns:
        intra_week_features.append(grouped['Power_GasShare'].apply(lambda x: calculate_slope(x.diff().dropna())).rename('Power_GasShare_acceleration'))
    
    # Inventory
    if 'Inventory_vs_5yrAvg' in df_daily_with_weeks.columns:
        intra_week_features.append(grouped['Inventory_vs_5yrAvg'].apply(calculate_slope).rename('Inventory_vs_5yrAvg_slope_7d'))
    
    if intra_week_features:
        df_intra_week = pd.concat(intra_week_features, axis=1).reset_index()
        print(f"    - Created {df_intra_week.shape[1] - 2} new intra-week features.")
    else:
        df_intra_week = pd.DataFrame(columns=['Week_Ending_Friday', 'Region']) # Empty df to avoid merge error
        print("    - No intra-week features were created.")


    # --- B. Standard Weekly Aggregation ---
    df_daily_with_weeks.columns = df_daily_with_weeks.columns.map(str)
    aggs = {col: 'sum' if any(x in col for x in ['HDD', 'CDD', 'Storage']) else ['mean', 'std'] for col in df_daily_with_weeks.columns if col not in ['Date', 'Region', 'Week_Ending_Friday']}
    
    print(f"  - Aggregating {len(aggs)} daily features into weekly stats...")
    df_weekly = df_daily_with_weeks.groupby(['Week_Ending_Friday', 'Region']).agg(aggs).reset_index()
    df_weekly.columns = ['_'.join(col).strip('_') for col in df_weekly.columns.values]
    df_weekly.rename(columns={'Week_Ending_Friday_': 'Week_Ending_Friday', 'Region_': 'Region'}, inplace=True)

    # --- C. Combine standard and intra-week features ---
    if not df_intra_week.empty:
        df_weekly = pd.merge(df_weekly, df_intra_week, on=['Week_Ending_Friday', 'Region'], how='left')
    print(f"    - Weekly aggregation complete. Shape: {df_weekly.shape}")

    # --- D. Lags and Deltas ---
    print("  - Creating lagged and momentum features...")
    df_weekly_final = df_weekly.copy()
    df_weekly_final.set_index(['Week_Ending_Friday', 'Region'], inplace=True)
    feature_cols = [col for col in df_weekly.columns if col not in ['Week_Ending_Friday', 'Region']]
    for lag in [1, 2, 4]:
        df_weekly_final = df_weekly_final.join(df_weekly_final.groupby('Region')[feature_cols].shift(lag).add_suffix(f'_lag_{lag}w'))
    for period in [1, 4]:
        df_weekly_final = df_weekly_final.join(df_weekly_final.groupby('Region')[feature_cols].diff(periods=period).add_suffix(f'_delta_{period}w'))
    df_weekly_final.reset_index(inplace=True)
    print(f"    - Final weekly feature set created with shape: {df_weekly_final.shape}")
    return df_weekly_final

# =============================================================================
# --- 5. MAIN EXECUTION ---
# =============================================================================
def main():
    """Main function to orchestrate the entire pipeline."""
    all_data = load_all_data(INFO_DIR)
    if all_data["eia_changes"].empty:
        raise FileNotFoundError("EIAchanges.csv is required but was not loaded.")

    df_daily_features = engineer_daily_features(all_data)
    df_weekly_features = aggregate_to_weekly(df_daily_features, all_data["eia_changes"])

    target_cols = [col for col in all_data["eia_changes"].columns if 'Storage Change' in col]
    df_targets = all_data["eia_changes"][['Period'] + target_cols]
    df_targets = df_targets.melt(id_vars='Period', var_name='Metric', value_name='Target_Storage_Change')
    df_targets['Region'] = df_targets['Metric'].str.split(' ').str[0]
    df_targets = df_targets.pivot_table(index='Period', columns='Region', values='Target_Storage_Change')
    df_targets.index.name = 'Week_Ending_Friday'
    df_targets_final = df_targets.stack().reset_index().rename(columns={'level_1': 'Region', 0: 'Target_Storage_Change'})
    
    df_final_dataset = pd.merge(df_weekly_features, df_targets_final, on=['Week_Ending_Friday', 'Region'], how='inner')
    
    print("\n--- Saving Final Feature Sets ---")
    # UPDATED: Using the requested hardcoded output path
    output_dir = Path('C:/Users/patri/OneDrive/Desktop/Coding/TraderHelper/EIAGuesser/output')
    output_dir.mkdir(exist_ok=True)
    
    master_path = output_dir / 'master_weekly_features.csv'
    df_final_dataset.to_csv(master_path, index=False)
    print(f"  - ✅ Master feature set saved to: {master_path}")
    
    for region, data in df_final_dataset.groupby('Region'):
        region_path = output_dir / f'features_{region.lower()}.csv'
        data.to_csv(region_path, index=False)
        print(f"  - ✅ {region} feature set saved to: {region_path}")
        
    print("\n--- Feature Engineering Pipeline Complete ---")

if __name__ == '__main__':
    main()