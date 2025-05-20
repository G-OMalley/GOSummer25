#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-
"""
Cleaned and Corrected Python Script for EIA Regional Forecasting.

This script performs the following steps:
1.  **MANUAL STEP**: Ensure necessary libraries are installed in your environment.
    Run: pip install optuna shap lightgbm statsmodels joblib pandas numpy openpyxl matplotlib
2.  Imports dependencies and sets up the environment.
3.  Loads and preprocesses input data (Fundamentals, Weather, EIA, PowerGen).
    **CRITICAL**: Data files (Fundy.csv, WEATHER.csv, EIAchanges.csv, PowerGen.xlsx)
    must be in the same directory as this script, or you MUST update the `data_dir` variable.
4.  Creates regional feature DataFrames by merging relevant data sources.
5.  Performs final data cleaning (date cutoff, interpolation).
6.  Sets correct DatetimeIndexes for all final DataFrames.
7.  Runs the hybrid LGBM+ARIMA forecasting pipeline for each region (generates a 1-week estimate).
8.  Displays results, saves models, and shows feature importance.
(Multi-week forecast functionality has been removed as per request).
"""

# === Cell 1: Install Libraries (User Action Required) ===
# Make sure you have installed the necessary libraries in your Python environment.
# Open your terminal and run (ideally in a virtual environment):
# pip install optuna shap lightgbm statsmodels joblib pandas numpy openpyxl matplotlib

# === Cell 2: Import Dependencies & Setup Environment ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from IPython.display import display # Using print() instead for script compatibility
import os
import warnings
import io
import traceback # For detailed error printing
import re
import logging
import dataclasses
import datetime
from typing import List, Dict, Any, Optional, Tuple

# Machine Learning & Stats Libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
import shap
import joblib
import optuna

# Google Colab specific code has been removed (drive mounting)

# --- Environment Setup ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=ValueWarning)
pd.set_option('display.max_columns', None)
optuna.logging.set_verbosity(optuna.logging.WARNING) # Suppress optuna logs

# --- Logging Setup ---
logger = logging.getLogger()
# Clear existing handlers if any (useful in interactive environments)
if logger.hasHandlers():
    logger.handlers.clear()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_step(name: str):
    """Logs a separator and step name."""
    logging.info(f"\n{'='*30} {name} {'='*30}")

# Google Drive mounting code has been removed.
# Data will be loaded from the local directory specified by `data_dir`.

# === Cell 3: Load and Preprocess Input Data ===
log_step("Loading and Preprocessing Input Data")

try:
    # Get the directory where the current script is located (EIAGuesser)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go one level up to the project's root directory (TraderHelper)
    project_root = os.path.dirname(script_dir)
    
    # Set the data directory to be the INFO folder inside the project root
    data_dir = os.path.join(project_root, 'INFO')
    
    logging.info(f"Data directory is set to: {data_dir}") # This will now log the correct INFO path
    logging.info("Please ensure Fundy.csv, WEATHER.csv, EIAchanges.csv, PowerGen.xlsx are in this directory.") # This log message is still relevant for the INFO dir
except Exception as e:
    # ... (the rest of your existing error handling for data_dir) ...
    logging.error(f"Could not set data_dir: {e}") # You might want to update this message slightly
    raise

# --- Load Core Files ---
try:
    logging.info("Loading Fundy.csv...")
    fundy_path = os.path.join(data_dir, 'Fundy.csv')
    if not os.path.exists(fundy_path):
        logging.error(f"Fundy.csv not found at {fundy_path}")
        raise FileNotFoundError(f"Fundy.csv not found at {fundy_path}")
    fundy = pd.read_csv(fundy_path, parse_dates=['Date'])

    logging.info("Loading WEATHER.csv...")
    weather_path = os.path.join(data_dir, 'WEATHER.csv')
    if not os.path.exists(weather_path):
        logging.error(f"WEATHER.csv not found at {weather_path}")
        raise FileNotFoundError(f"WEATHER.csv not found at {weather_path}")
    weather_raw = pd.read_csv(weather_path, parse_dates=['Date'])

    logging.info("Loading EIAchanges.csv...")
    eia_changes_path = os.path.join(data_dir, 'EIAchanges.csv')
    if not os.path.exists(eia_changes_path):
        logging.error(f"EIAchanges.csv not found at {eia_changes_path}")
        raise FileNotFoundError(f"EIAchanges.csv not found at {eia_changes_path}")
    eia_changes = pd.read_csv(eia_changes_path, parse_dates=['Week ending'])

    logging.info("Loading PowerGen.xlsx...")
    powergen_path = os.path.join(data_dir, 'PowerGen.xlsx')
    if not os.path.exists(powergen_path):
        logging.error(f"PowerGen.xlsx not found at {powergen_path}")
        raise FileNotFoundError(f"PowerGen.xlsx not found at {powergen_path}")
    powergen_raw = pd.read_excel(powergen_path, sheet_name=None)

except FileNotFoundError:
    logging.error(f"CRITICAL ERROR: One or more input files were not found. Please check the path: {data_dir}.")
    logging.error("Ensure your data files are in the same directory as the script, or update the 'data_dir' variable.")
    raise
except Exception as e:
    logging.error(f"ERROR loading input files: {e}")
    raise

# --- Preprocess Fundy ---
log_step("Pivoting Fundy Data")
try:
    fundy_wide = fundy.pivot(index='Date', columns='item', values='value').reset_index()
    logging.info(f"Fundy pivoted. Shape: {fundy_wide.shape}")
    fundy_wide['Date'] = pd.to_datetime(fundy_wide['Date'])
except Exception as e:
    logging.error(f"ERROR pivoting Fundy data: {e}")
    raise

# --- Preprocess Weather ---
log_step("Pivoting Weather Data")
try:
    weather_processed = weather_raw.drop(columns=['City Symbol'], errors='ignore')
    weather_melted = weather_processed.melt(id_vars=['Date', 'City Title'], var_name='Metric', value_name='Value')
    weather_melted['Column'] = weather_melted['City Title'] + ' - ' + weather_melted['Metric']
    weather_wide = weather_melted.pivot(index='Date', columns='Column', values='Value').reset_index()
    logging.info(f"Weather pivoted. Shape: {weather_wide.shape}")
    weather_wide['Date'] = pd.to_datetime(weather_wide['Date'])
except Exception as e:
    logging.error(f"ERROR processing Weather data: {e}")
    raise

# --- Preprocess PowerGen ---
log_step("Processing PowerGen Data")
powergen = {}
try:
    powergen_raw.pop('M2MS Prompt Prices', None)
    for tab_name, df_raw in powergen_raw.items():
        df = df_raw.copy()
        if 'Date' not in df.columns:
            logging.warning(f"Skipping PowerGen tab '{tab_name}': Missing 'Date' column.")
            continue
        df['Date'] = pd.to_datetime(df['Date'])
        new_cols = {'Date': 'Date'}
        for col in df.columns:
            if col != 'Date':
                prefix = f"{tab_name} - "
                new_col_name = prefix + col if not str(col).startswith(prefix) else str(col)
                new_cols[col] = new_col_name
        df.rename(columns=new_cols, inplace=True)
        powergen[tab_name] = df
        logging.info(f"Processed PowerGen tab: '{tab_name}'. Columns: {df.columns.tolist()[:5]}...")
except Exception as e:
    logging.error(f"ERROR processing PowerGen data: {e}")
    raise

# === Cell 4: Construct Regional Feature DataFrames ===
log_step("Constructing Regional Feature DataFrames")

if 'Date' not in fundy_wide.columns:
    raise ValueError("Critical Error: 'Date' column not found in fundy_wide after pivoting.")
daily_dates = fundy_wide['Date'].unique()
base_df_template = pd.DataFrame({'Date': sorted(daily_dates)})

total_df = base_df_template.copy()
east_df = base_df_template.copy()
midwest_df = base_df_template.copy()
mountain_df = base_df_template.copy()
pacific_df = base_df_template.copy()
south_central_df = base_df_template.copy()

regional_dfs = {
    'total': total_df, 'east': east_df, 'midwest': midwest_df,
    'mountain': mountain_df, 'pacific': pacific_df, 'south_central': south_central_df
}

log_step("Assigning Fundy Columns to Regional Tables")
try:
    regional_dfs['total'] = pd.merge(regional_dfs['total'], fundy_wide, on='Date', how='left')
    logging.info(f"Merged all Fundy columns into total_df. Shape: {regional_dfs['total'].shape}")
    conus_cols = [col for col in fundy_wide.columns if str(col).startswith('CONUS') or str(col) == 'GOM - Prod']
    fundy_col_map = {
        'midwest': conus_cols + ['Mid West - Balance', 'Midwest - Ind', 'Midwest - Power', 'Midwest - Prod', 'Midwest - ResCom'],
        'east': conus_cols + ['North East - Balance', 'Northeast - Ind', 'Northeast - Power', 'Northeast - Prod', 'Northeast - ResCom', 'South East - Balance', 'SouthEast - Ind', 'SouthEast - Power', 'SouthEast - ResCom', 'SouthEast[Fl] - Balance', 'SouthEast[Fl] - Ind', 'SouthEast[Fl] - Power', 'SouthEast[Fl] - ResCom', 'SouthEast[Oth] - Balance', 'SouthEast[Oth] - Ind', 'SouthEast[Oth] - Power', 'SouthEast[Oth] - ResCom', 'Southeast - Prod'],
        'mountain': conus_cols + ['Rockies - Balance', 'Rockies - Ind', 'Rockies - Power', 'Rockies - Prod', 'Rockies - ResCom', 'Rockies[SW] - Balance', 'Rockies[SW] - Ind', 'Rockies[SW] - Power', 'Rockies[SW] - ResCom', 'Rockies[Up] - Balance', 'Rockies[Up] - Ind', 'Rockies[Up] - Power', 'Rockies[Up] - ResCom'],
        'south_central': conus_cols + ['South Central - Balance', 'SouthCentral - Ind', 'SouthCentral - Power', 'SouthCentral - Prod', 'SouthCentral - ResCom'],
        'pacific': conus_cols + ['West - Balance', 'West - Prod', 'West[CA] - Balance', 'West[CA] - Ind', 'West[CA] - Power', 'West[CA] - ResCom', 'West[PNW] - Balance', 'West[PNW] - Ind', 'West[PNW] - Power', 'West[PNW] - ResCom']
    }
    for region, df_iter in regional_dfs.items(): 
        if region != 'total':
            if region in fundy_col_map:
                cols_to_merge = ['Date'] + [col for col in fundy_col_map[region] if col in fundy_wide.columns]
                cols_index = pd.Index(cols_to_merge)
                fundy_subset = fundy_wide[cols_index.unique()]
                regional_dfs[region] = pd.merge(df_iter, fundy_subset, on='Date', how='left')
                logging.info(f"Merged Fundy columns into {region}_df. Shape: {regional_dfs[region].shape}")
            else:
                logging.warning(f"Region key '{region}' not found in fundy_col_map. Skipping Fundy merge.")
except KeyError as e:
    logging.error(f"KeyError merging Fundy data: Column '{e}' not found. Check names.")
    raise
except Exception as e:
    logging.error(f"ERROR assigning Fundy columns: {e}")
    raise

log_step("Assigning Weather Columns to Regional Tables")
try:
    city_region_map = {
        'East Region': ['Atlanta GA', 'Boston MA', 'Buffalo NY', 'John F. Kennedy NY', 'Philadelphia PA', 'Pittsburgh PA', 'Raleigh/Durham NC', 'Tampa FL', 'Washington National DC'],
        'Midwest Region': ['Chicago IL', 'Detroit MI'],
        'Mountain Region': ['Denver CO'],
        'South Central Region': ['Houston TX', 'Little Rock AR', 'New Orleans LA', 'Oklahoma City OK'],
        'Pacific Region': ['Los Angeles CA', 'San Francisco CA', 'Seattle WA']
    }
    weather_regional = {}
    for region_full_name, cities in city_region_map.items():
        region_key = 'south_central' if region_full_name == 'South Central Region' else region_full_name.split(' ')[0].lower()
        if region_key not in regional_dfs:
            logging.warning(f"Weather region key '{region_key}' mismatch. Skipping for {region_full_name}.")
            continue
        cols_to_select = ['Date'] + [col for col in weather_wide.columns if any(str(col).startswith(city) for city in cities)]
        cols_index = pd.Index(cols_to_select)
        weather_regional[region_key] = weather_wide[cols_index.unique()]
        logging.info(f"Created weather_{region_key} DataFrame. Shape: {weather_regional[region_key].shape}")

    regional_dfs['total'] = pd.merge(regional_dfs['total'], weather_wide, on='Date', how='left')
    logging.info(f"Merged all Weather columns into total_df. Shape: {regional_dfs['total'].shape}")
    for region_key, weather_df in weather_regional.items():
        regional_dfs[region_key] = pd.merge(regional_dfs[region_key], weather_df, on='Date', how='left')
        logging.info(f"Merged Weather columns into {region_key}_df. Shape: {regional_dfs[region_key].shape}")
except Exception as e:
    logging.error(f"ERROR assigning Weather columns: {e}")
    raise

log_step("Assigning PowerGen Columns to Regional Tables")
try:
    power_col_map = {
        'CAISO Generation': {'Pacific': ['CAISO Generation - Total Generation', 'CAISO Generation - Thermal Power', 'CAISO Generation - Thermal Share'], 'Total': ['CAISO Generation - Total Generation', 'CAISO Generation - Thermal Power', 'CAISO Generation - Thermal Share']},
        'ERCOT Supply': {'South Central': ['ERCOT Supply - Total Generation', 'ERCOT Supply - Natural Gas'], 'Total': ['ERCOT Supply - Total Generation', 'ERCOT Supply - Natural Gas']},
        'MISO Generation': {'Midwest': ['MISO Generation - Total Generation', 'MISO Generation - Natural Gas', 'MISO Generation - Gas Share'], 'South Central': ['MISO Generation - Total Generation', 'MISO Generation - Natural Gas', 'MISO Generation - Gas Share'], 'Total': ['MISO Generation - Total Generation', 'MISO Generation - Natural Gas', 'MISO Generation - Gas Share']},
        'ISONE Generation': {'East': ['ISONE Generation - Total Generation', 'ISONE Generation - Natural Gas', 'ISONE Generation - Gas Share'], 'Total': ['ISONE Generation - Total Generation', 'ISONE Generation - Natural Gas', 'ISONE Generation - Gas Share']},
        'NYISO Generation': {'East': ['NYISO Generation - Total Generation', 'NYISO Generation - Natural Gas', 'NYISO Generation - Dual Fuel', 'NYISO Generation - Dual Fuel Share', 'NYISO Generation - Gas Share'], 'Total': ['NYISO Generation - Total Generation', 'NYISO Generation - Natural Gas', 'NYISO Generation - Dual Fuel', 'NYISO Generation - Dual Fuel Share', 'NYISO Generation - Gas Share']},
        'PJM Generation': {'East': ['PJM Generation - Total Generation', 'PJM Generation - Natural Gas', 'PJM Generation - Gas Share'], 'Total': ['PJM Generation - Total Generation', 'PJM Generation - Natural Gas', 'PJM Generation - Gas Share']},
        'SPP Generation': {'Midwest': ['SPP Generation - Total Generation', 'SPP Generation - Natural Gas', 'SPP Generation - Gas Share'], 'South Central': ['SPP Generation - Total Generation', 'SPP Generation - Natural Gas', 'SPP Generation - Gas Share'], 'Total': ['SPP Generation - Total Generation', 'SPP Generation - Natural Gas', 'SPP Generation - Gas Share']}
    }
    power_regional_dfs = {region_key: [] for region_key in regional_dfs.keys()}
    for tab, mappings in power_col_map.items():
        if tab in powergen:
            df_power_tab = powergen[tab]
            for region_full, cols_to_select in mappings.items():
                if region_full == 'South Central': region_key = 'south_central'
                elif region_full == 'Total': region_key = 'total'
                else: region_key = region_full.split(' ')[0].lower()
                if region_key not in power_regional_dfs:
                    logging.warning(f"Power region key '{region_key}' not in keys. Skipping.")
                    continue
                valid_cols = ['Date'] + [c for c in cols_to_select if c in df_power_tab.columns]
                if len(valid_cols) > 1:
                    cols_index = pd.Index(valid_cols)
                    power_regional_dfs[region_key].append(df_power_tab[cols_index.unique()])
                else:
                    logging.warning(f"No valid columns for PowerGen tab '{tab}' in region '{region_full}'.")
        else:
            logging.warning(f"PowerGen tab '{tab}' not found in loaded data.")

    final_power_dfs = {}
    for region_key, df_list in power_regional_dfs.items():
        if df_list:
            merged_df = df_list[0]
            for i in range(1, len(df_list)):
                cols_to_merge = df_list[i].columns.difference(['Date'])
                cols_in_merged = merged_df.columns.difference(['Date'])
                duplicates = cols_to_merge.intersection(cols_in_merged)
                if not duplicates.empty:
                    logging.warning(f"Duplicate columns for power data region '{region_key}': {duplicates.tolist()}.")
                    merged_df = pd.merge(merged_df, df_list[i], on='Date', how='outer', suffixes=('', '_dup'))
                    merged_df = merged_df[[col for col in merged_df.columns if not str(col).endswith('_dup')]]
                else:
                    merged_df = pd.merge(merged_df, df_list[i], on='Date', how='outer')
            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
            final_power_dfs[region_key] = merged_df
            logging.info(f"Combined power DataFrame for '{region_key}'. Shape: {merged_df.shape}")
        else:
            logging.info(f"No power data for region '{region_key}'.")

    for region_key, power_df in final_power_dfs.items():
        if region_key in regional_dfs:
            if power_df is not None and not power_df.empty:
                regional_dfs[region_key] = pd.merge(regional_dfs[region_key], power_df, on='Date', how='left')
                logging.info(f"Merged PowerGen columns into {region_key}_df. Shape: {regional_dfs[region_key].shape}")
            else:
                logging.info(f"Skipping PowerGen merge for '{region_key}' (no data).")
        else:
            logging.warning(f"Region key '{region_key}' from power data not in regional_dfs.")
except Exception as e:
    logging.error(f"ERROR assigning PowerGen columns: {e}")
    traceback.print_exc()
    raise

final_total_df = regional_dfs['total']
final_east_df = regional_dfs['east']
final_midwest_df = regional_dfs['midwest']
final_mountain_df = regional_dfs['mountain']
final_pacific_df = regional_dfs['pacific']
final_south_central_df = regional_dfs['south_central']

# === Cell 5: Final Data Cleaning and Index Setting ===
log_step("Final Data Cleaning and Index Setting")
cutoff_date = pd.to_datetime("2018-07-07")
logging.info(f"Applying date cutoff: >= {cutoff_date.date()}")
final_df_names_map = {
    'final_total_df': final_total_df, 'final_east_df': final_east_df,
    'final_midwest_df': final_midwest_df, 'final_mountain_df': final_mountain_df,
    'final_pacific_df': final_pacific_df, 'final_south_central_df': final_south_central_df
}
for df_name, df_obj in final_df_names_map.items():
    if df_obj is None or df_obj.empty:
        logging.warning(f"{df_name} is None/empty before cutoff. Skipping.")
        continue
    original_rows = len(df_obj)
    if 'Date' not in df_obj.columns:
        logging.error(f"'Date' column missing in {df_name} before cutoff.")
        continue
    df_filtered = df_obj[df_obj['Date'] >= cutoff_date].copy()
    rows_dropped = original_rows - len(df_filtered)
    globals()[df_name] = df_filtered
    logging.info(f"Cutoff {df_name}. Dropped: {rows_dropped}. New shape: {df_filtered.shape}")

log_step("Interpolating and Filling Missing Values")
for df_name in final_df_names_map.keys():
    df_iter = globals()[df_name] 
    logging.info(f"Processing {df_name}...")
    if df_iter is None or df_iter.empty:
        logging.warning(f"{df_name} is None/empty. Skipping interpolation.")
        continue
    if 'Date' not in df_iter.columns:
        logging.error(f"'Date' missing in {df_name} before interpolation.")
        continue
    df_iter['Date'] = pd.to_datetime(df_iter['Date'])
    df_iter.sort_values("Date", inplace=True)
    numeric_cols = df_iter.select_dtypes(include=np.number).columns
    logging.info(f"  Interpolating {len(numeric_cols)} numeric columns...")
    df_iter[numeric_cols] = df_iter[numeric_cols].interpolate(method='linear', limit=1, limit_direction='both')
    df_iter[numeric_cols] = df_iter[numeric_cols].ffill()
    df_iter[numeric_cols] = df_iter[numeric_cols].bfill()
    nan_check = df_iter[numeric_cols].isnull().sum()
    remaining_nans = nan_check[nan_check > 0]
    if not remaining_nans.empty:
        logging.warning(f"  NaNs remain in {df_name}: {remaining_nans.index.tolist()}")
    else:
        logging.info(f"  No NaNs remaining in numeric columns for {df_name}.")

log_step("Setting Final Indexes")
for df_name in final_df_names_map.keys():
    df_iter = globals()[df_name] 
    if df_iter is None or df_iter.empty: continue
    if 'Date' in df_iter.columns:
        try:
            df_iter['Date'] = pd.to_datetime(df_iter['Date'])
            df_iter.set_index('Date', inplace=True)
            logging.info(f"Set 'Date' as DatetimeIndex for {df_name}.")
            inferred_freq = pd.infer_freq(df_iter.index)
            if inferred_freq != 'D':
                logging.warning(f"Freq for {df_name} inferred as '{inferred_freq}', expected 'D'.")
            else:
                logging.info(f"Verified freq for {df_name} as 'D'.")
        except Exception as e:
            logging.error(f"Failed to set 'Date' index for {df_name}: {e}")
            raise
    elif isinstance(df_iter.index, pd.DatetimeIndex):
        logging.info(f"{df_name} already has DatetimeIndex.")
    else:
        logging.error(f"'Date' column not found for {df_name}.")
        raise TypeError(f"Cannot set index for {df_name}")

if 'Week ending' in eia_changes.columns:
    try:
        eia_changes['Week ending'] = pd.to_datetime(eia_changes['Week ending'])
        eia_changes.set_index('Week ending', inplace=True)
        logging.info("Set 'Week ending' as DatetimeIndex for eia_changes.")
        inferred_freq = pd.infer_freq(eia_changes.index)
        if inferred_freq != 'W-FRI':
            logging.warning(f"Freq for eia_changes: '{inferred_freq}', expected 'W-FRI'.")
            try:
                eia_changes = eia_changes.asfreq('W-FRI')
                logging.info("Set frequency for eia_changes to 'W-FRI'.")
            except ValueError as ve:
                logging.error(f"Could not set freq for eia_changes: {ve}")
        else:
            logging.info("Verified freq for eia_changes as 'W-FRI'.")
    except Exception as e:
        logging.error(f"Failed to set 'Week ending' index for eia_changes: {e}")
        raise
elif isinstance(eia_changes.index, pd.DatetimeIndex):
    logging.info("eia_changes already has DatetimeIndex.")
    inferred_freq = pd.infer_freq(eia_changes.index) 
    if inferred_freq != 'W-FRI':
        logging.warning(f"eia_changes DatetimeIndex, but freq '{inferred_freq}' != 'W-FRI'.")
        try:
            eia_changes = eia_changes.asfreq('W-FRI')
            logging.info("Set frequency for existing eia_changes index to 'W-FRI'.")
        except ValueError as ve:
            logging.error(f"Could not set freq for existing eia_changes index: {ve}")
    else:
        logging.info("Verified frequency for eia_changes as 'W-FRI'.")

else:
    logging.error("'Week ending' not found for eia_changes.")
    raise TypeError("Cannot set index for eia_changes")

TARGET_FEATURE_MAP_CHECK = {
    'East Region': 'final_east_df', 'Midwest Region': 'final_midwest_df',
    'Mountain Region': 'final_mountain_df', 'Pacific Region': 'final_pacific_df',
    'South Central Region': 'final_south_central_df', 'Total Lower 48': 'final_total_df',
}
target_cols_to_check = list(TARGET_FEATURE_MAP_CHECK.keys())
missing_targets = [col for col in target_cols_to_check if col not in eia_changes.columns]
if missing_targets:
    logging.error(f"CRITICAL: Target columns missing from eia_changes: {missing_targets}")
    logging.error(f"Available columns in eia_changes: {eia_changes.columns.tolist()}")
    raise ValueError("Target columns missing in eia_changes.")
else:
    logging.info("Verified target columns exist in final eia_changes.")

log_step("Data Preparation Complete")
print("\nfinal_total_df head (Index: Date):")
if 'final_total_df' in globals() and final_total_df is not None and not final_total_df.empty: print(final_total_df.head(2))
else: print("final_total_df is empty or not defined.")
print("\neia_changes head (Index: Week ending):")
if 'eia_changes' in globals() and eia_changes is not None and not eia_changes.empty: print(eia_changes.head(2))
else: print("eia_changes is empty or not defined.")


# === Cell 6: Forecasting Pipeline ===
@dataclasses.dataclass
class PipelineConfig:
    optuna_trials: int = 40; cv_splits: int = 5; random_state: int = 42
    target_lags: List[int] = dataclasses.field(default_factory=lambda: [1,2,3,4,8,13,26,52])
    rolling_windows_diff: List[int] = dataclasses.field(default_factory=lambda: [4,8,13])
    rolling_windows_level: List[int] = dataclasses.field(default_factory=lambda: [4,8,13,52])
    feature_importance_top_n: int = 15
    ljung_box_lags: List[int] = dataclasses.field(default_factory=lambda: [10])
    use_hybrid_arima: bool = True; arima_order: Tuple[int,int,int] = (2,0,1)

TARGET_FEATURE_MAP = {
    'East Region': 'final_east_df', 'Midwest Region': 'final_midwest_df',
    'Mountain Region': 'final_mountain_df', 'Pacific Region': 'final_pacific_df',
    'South Central Region': 'final_south_central_df', 'Total Lower 48': 'final_total_df',
}

log_step("Defining and Verifying Required Variables for Pipeline")
dataframe_vars = {}; target_map = {}
try:
    required_df_names = list(TARGET_FEATURE_MAP.values()) + ['eia_changes']
    for df_name in required_df_names:
        if df_name not in globals() or globals()[df_name] is None:
            raise NameError(f"DataFrame '{df_name}' not found or is None.")
        df_copy = globals()[df_name].copy()
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            raise TypeError(f"'{df_name}' no DatetimeIndex for pipeline.")
        logging.info(f"Verified DatetimeIndex for '{df_name}'.")
        target_freq = 'W-FRI' if df_name == 'eia_changes' else 'D'
        inferred_freq = pd.infer_freq(df_copy.index)
        if df_name != 'eia_changes' and inferred_freq is None:
            logging.warning(f"Freq for '{df_name}' not inferred (gaps likely).")
        elif inferred_freq != target_freq:
            logging.warning(f"Freq mismatch for '{df_name}': Inf='{inferred_freq}', Exp='{target_freq}'.")
        else:
            logging.info(f"Verified freq '{target_freq}' for '{df_name}'.")
        dataframe_vars[df_name] = df_copy
    logging.info("Created 'dataframe_vars' for pipeline.")
    target_map = {target_name: target_name for target_name in TARGET_FEATURE_MAP.keys()}
    logging.info("Created 'target_map' for pipeline.")
    eia_df_pipeline = dataframe_vars['eia_changes'] 
    missing_cols = [col for col in target_map.values() if col not in eia_df_pipeline.columns]
    if missing_cols:
        raise ValueError(f"Targets MISSING in eia_changes for pipeline: {missing_cols}")
    logging.info("All target columns verified in eia_changes for pipeline.")
except (NameError, TypeError, ValueError) as e:
    logging.error(f"Pipeline variable setup failed: {e}"); raise
except Exception as e:
    logging.error(f"Unexpected error in pipeline setup: {e}"); raise

def sanitize_feature_name(name):
    name = str(name).replace('/','_').replace('-','_').replace(' ','_').replace('[','_').replace(']','_').replace('(','_').replace(')','_').replace('.','_')
    name = re.sub(r'[^A-Za-z0-9_]+', '', name)
    if name and name[0].isdigit(): name = '_' + name
    if not name: name = 'sanitized_empty_col'
    return name

def _fit_arima_residuals(residuals: pd.Series, order: Tuple[int,int,int]):
    log_step(f"Fitting ARIMA{order} to Residuals")
    residuals_clean = residuals.dropna()
    if residuals_clean.empty: logging.warning("  Residual series empty."); return None
    try:
        if residuals_clean.index.freq is None:
            freq = pd.infer_freq(residuals_clean.index)
            if freq == 'W-FRI': residuals_clean = residuals_clean.asfreq(freq); logging.info(f"  Inferred/set freq '{freq}'.")
            elif freq:
                logging.warning(f" Inferred freq '{freq}' != 'W-FRI'. Attempting 'W-FRI'.")
                try: residuals_clean = residuals_clean.asfreq('W-FRI'); logging.info("    Set freq to 'W-FRI'.")
                except ValueError: logging.error("    Failed to set 'W-FRI'. ARIMA may fail."); return None
            else: logging.error("  Cannot determine freq for ARIMA. Aborted."); return None
        model = ARIMA(residuals_clean, order=order, enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(); logging.info(f"  ARIMA fitting complete. LL: {results.llf:.2f}"); return results
    except ValueError as ve: logging.error(f"  ValueError ARIMA{order}: {ve}."); return None
    except Exception as e: logging.error(f"  Error fitting ARIMA{order}: {e}"); return None

def _run_single_target_pipeline(target_col_name_original: str, feature_df_name: str,
                                all_data_vars: Dict[str, pd.DataFrame], config: PipelineConfig) -> Optional[Dict[str, Any]]:
    log_step(f"Pipeline: Target='{target_col_name_original}', Features='{feature_df_name}'")
    if config.use_hybrid_arima: logging.info("Mode: Hybrid LGBM + ARIMA")
    else: logging.info("Mode: Standard LGBM")

    target_df_name_pipeline = 'eia_changes' 
    feature_df_pipeline = all_data_vars[feature_df_name] 
    target_df_pipeline = all_data_vars[target_df_name_pipeline] 

    if target_col_name_original not in target_df_pipeline.columns: logging.error(f"Target '{target_col_name_original}' missing."); return None
    if not isinstance(feature_df_pipeline.index, pd.DatetimeIndex): logging.error(f"Feature DF '{feature_df_name}' index error."); return None
    if not isinstance(target_df_pipeline.index, pd.DatetimeIndex): logging.error(f"Target DF '{target_df_name_pipeline}' index error."); return None
    if feature_df_pipeline.empty: logging.error(f"Feature DF '{feature_df_name}' empty."); return None

    log_step("Step 1: Aggregating features (W-FRI)")
    df_numeric = feature_df_pipeline.select_dtypes(include=np.number)
    if df_numeric.empty: logging.error(f"No numeric cols in '{feature_df_name}'."); return None
    agg_methods = {}; sanitized_agg_col_map = {}
    for col in df_numeric.columns:
        col_lower = str(col).lower()
        if any(k in col_lower for k in ['price','temp','avg','pct','share','hdd','cdd']): agg_methods[col]='mean'
        elif any(k in col_lower for k in ['storage','level','inventory']): agg_methods[col]='last'
        else: agg_methods[col]='sum'
        sanitized_agg_col_map[col] = sanitize_feature_name(f"Agg_{col}")
    try:
        df_numeric_sorted = df_numeric.sort_index() 
        if not df_numeric_sorted.index.is_unique:
            logging.warning(f"  Duplicate dates in {feature_df_name}. Aggregating by mean.")
            df_numeric_sorted = df_numeric_sorted.groupby(df_numeric_sorted.index).mean()
        X_weekly_aggregated = df_numeric_sorted.resample('W-FRI').agg(agg_methods)
        logging.info(f"  - Aggregated weekly shape: {X_weekly_aggregated.shape}")
        new_agg_cols = [sanitized_agg_col_map[col] for col in X_weekly_aggregated.columns]
        if len(new_agg_cols) != len(set(new_agg_cols)):
            logging.warning("Duplicate sanitized col names post-agg. Adding suffixes.")
            counts = {}; final_agg_cols = []
            for col_name in new_agg_cols:
                if col_name in counts: counts[col_name]+=1; final_agg_cols.append(f"{col_name}_{counts[col_name]}")
                else: counts[col_name]=0; final_agg_cols.append(col_name)
            X_weekly_aggregated.columns = final_agg_cols
        else: X_weekly_aggregated.columns = new_agg_cols
        logging.info(f"  - Renamed aggregated columns.")
    except Exception as e: logging.error(f"ERROR aggregating: {e}"); traceback.print_exc(); return None

    log_step("Step 2: Preparing Target and Base Features")
    y_weekly = target_df_pipeline[[target_col_name_original]].copy()
    y_weekly = y_weekly.sort_index()
    try:
        target_freq = pd.infer_freq(y_weekly.index)
        if target_freq != 'W-FRI':
            logging.warning(f"Target freq '{target_freq}' != 'W-FRI'. Resampling.")
            y_weekly = y_weekly.resample('W-FRI').mean()
            logging.info(f"  - Resampled target shape: {y_weekly.shape}")
        else: logging.info(f"  - Verified target freq: W-FRI")
    except Exception as e: logging.warning(f"  Could not verify/resample target freq: {e}")

    base_features = pd.DataFrame(index=y_weekly.index)
    sanitized_target_name_for_features = sanitize_feature_name(target_col_name_original)
    logging.info(f"  - Creating lag features...")
    for lag in config.target_lags:
        base_features[sanitize_feature_name(f'{sanitized_target_name_for_features}_lag_{lag}')] = y_weekly[target_col_name_original].shift(lag)
    target_diff = y_weekly[target_col_name_original].diff()
    for lag in config.target_lags:
        base_features[sanitize_feature_name(f'{sanitized_target_name_for_features}_diff_lag_{lag}')] = target_diff.shift(lag)

    logging.info("  - Creating rolling features...")
    diff_lag1_feat_name = sanitize_feature_name(f'{sanitized_target_name_for_features}_diff_lag_1')
    if diff_lag1_feat_name in base_features.columns:
        for window in config.rolling_windows_diff:
            base_features[sanitize_feature_name(f'{diff_lag1_feat_name}_roll_mean_{window}')] = base_features[diff_lag1_feat_name].rolling(window,min_periods=1).mean()
            base_features[sanitize_feature_name(f'{diff_lag1_feat_name}_roll_std_{window}')] = base_features[diff_lag1_feat_name].rolling(window,min_periods=1).std()
    lag1_feat_name = sanitize_feature_name(f'{sanitized_target_name_for_features}_lag_1')
    if lag1_feat_name in base_features.columns:
        for window in config.rolling_windows_level:
            base_features[sanitize_feature_name(f'{lag1_feat_name}_roll_std_{window}')] = base_features[lag1_feat_name].rolling(window,min_periods=1).std()

    logging.info("  - Creating calendar features...");
    base_features[sanitize_feature_name('cal_weekofyear')] = base_features.index.isocalendar().week.astype(int)
    base_features[sanitize_feature_name('cal_month')] = base_features.index.month
    base_features[sanitize_feature_name('cal_year')] = base_features.index.year
    std_dev_cols = [col for col in base_features.columns if 'roll_std' in col]
    base_features[std_dev_cols] = base_features[std_dev_cols].fillna(0)

    log_step("Step 3: Combining Features and Alignment")
    X_combined_initial = base_features.join(X_weekly_aggregated, how='left')
    y_aligned_initial = y_weekly.loc[X_combined_initial.index]
    logging.info(f"  - Combined shape before NaN drop: {X_combined_initial.shape}")
    if X_combined_initial.empty: logging.error("Combine resulted in empty DF."); return None
    const_cols = X_combined_initial.columns[X_combined_initial.nunique(dropna=False) <= 1].tolist()
    if const_cols: logging.info(f"  - Dropping constant: {const_cols}"); X_combined_initial = X_combined_initial.drop(columns=const_cols)
    logging.info(f"  - Initial NaNs (Top 10):\n{X_combined_initial.isnull().sum().sort_values(ascending=False).head(10)}")
    logging.info("  - Dropping rows with any NaNs...")
    initial_rows = X_combined_initial.shape[0]
    X_base = X_combined_initial.dropna(axis=0)
    y_base = y_aligned_initial.loc[X_base.index]
    logging.info(f"  - Dropped {initial_rows - X_base.shape[0]} rows.")
    logging.info(f"  - Base shapes: X={X_base.shape}, y={y_base.shape}")
    if X_base.empty or y_base.empty: logging.error("Base set empty after NaN handling."); return None
    logging.info(f"  - Sample final features: {X_base.columns[:5].tolist()}...")

    log_step("Step 4: Tuning & Training Base LGBM Model")
    def objective(trial, X, y, n_splits):
        y_np = y.values.ravel()
        params = {'objective':'regression_l1','metric':'mae','boosting_type':'gbdt',
                  'n_estimators':trial.suggest_int('n_estimators',200,1500,step=100),
                  'learning_rate':trial.suggest_float('learning_rate',0.005,0.1,log=True),
                  'num_leaves':trial.suggest_int('num_leaves',10,max(30,int(X.shape[1]*0.6))),
                  'max_depth':trial.suggest_int('max_depth',4,20),
                  'reg_alpha':trial.suggest_float('reg_alpha',1e-8,20.0,log=True),
                  'reg_lambda':trial.suggest_float('reg_lambda',1e-8,20.0,log=True),
                  'colsample_bytree':trial.suggest_float('colsample_bytree',0.4,1.0),
                  'subsample':trial.suggest_float('subsample',0.4,1.0),
                  'min_child_samples':trial.suggest_int('min_child_samples',5,60),
                  'random_state':config.random_state,'n_jobs':-1,'verbose':-1}
        tscv = TimeSeriesSplit(n_splits=n_splits); mae_scores = []; oof_preds_trial = pd.Series(index=X.index,dtype=float)
        try:
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_tr,X_v=X.iloc[train_idx],X.iloc[val_idx]; y_tr,y_v=y_np[train_idx],y_np[val_idx]
                model = lgb.LGBMRegressor(**params)
                model.fit(X_tr,y_tr,eval_set=[(X_v,y_v)],callbacks=[lgb.early_stopping(15,verbose=False)])
                preds = model.predict(X_v); oof_preds_trial.iloc[val_idx]=preds
                mae_scores.append(mean_absolute_error(y_v,preds))
            avg_mae = np.mean(mae_scores)
            if np.isnan(avg_mae) or np.isinf(avg_mae): logging.warning(f"Trial {trial.number} invalid MAE: {avg_mae}"); return float('inf')
            trial.set_user_attr("oof_predictions",oof_preds_trial); return avg_mae
        except Exception as e: logging.warning(f"Optuna trial failed: {e}"); traceback.print_exc(); return float('inf')

    study_base = optuna.create_study(direction='minimize')
    logging.info(f"  - Running Optuna (Base) for {config.optuna_trials} trials...")
    study_base.optimize(lambda trial: objective(trial,X_base,y_base[target_col_name_original],config.cv_splits),
                        n_trials=config.optuna_trials,show_progress_bar=False)
    if not study_base.best_trial: logging.error("Optuna no best trial."); return None
    best_params_base = study_base.best_params; best_mae_base = study_base.best_value
    logging.info(f"  - Optuna (Base) Complete. Best MAE: {best_mae_base:.4f}")
    logging.info("  - Retrieving OOF predictions (Base)..."); oof_predictions_base = None
    try:
        oof_predictions_base = study_base.best_trial.user_attrs.get("oof_predictions")
        if oof_predictions_base is None:
            logging.warning("OOF not in attrs. Recalculating...")
            oof_predictions_base = pd.Series(index=X_base.index,dtype=float)
            tscv = TimeSeriesSplit(n_splits=config.cv_splits)
            y_base_np = y_base[target_col_name_original].values.ravel()
            for fold, (train_idx,val_idx) in enumerate(tscv.split(X_base)):
                X_tr,X_v=X_base.iloc[train_idx],X_base.iloc[val_idx]; y_tr,y_v=y_base_np[train_idx],y_base_np[val_idx]
                model=lgb.LGBMRegressor(objective='regression_l1',metric='mae',random_state=config.random_state,n_jobs=-1,**best_params_base)
                model.fit(X_tr,y_tr,eval_set=[(X_v,y_v)],callbacks=[lgb.early_stopping(15,verbose=False)])
                oof_predictions_base.iloc[val_idx] = model.predict(X_v)
    except Exception as e: logging.error(f"ERROR OOF (Base): {e}")
    if oof_predictions_base is None or oof_predictions_base.isnull().all(): logging.error("No valid Base OOF."); return None

    lgbm_model = lgb.LGBMRegressor(objective='regression_l1',metric='mae',random_state=config.random_state,n_jobs=-1,**best_params_base)
    lgbm_model.fit(X_base,y_base[target_col_name_original].values.ravel())
    logging.info("  - Final Base LGBM model trained.")
    residuals_base = y_base[target_col_name_original] - oof_predictions_base
    logging.info(f"  - Base Residuals. Shape: {residuals_base.shape}, NaNs: {residuals_base.isnull().sum()}")

    arima_results = None; oof_predictions_final = oof_predictions_base.copy(); final_mae = best_mae_base
    if config.use_hybrid_arima:
        log_step("Step 5: Fitting ARIMA on Base Model Residuals")
        residuals_for_arima = residuals_base.dropna()
        arima_results = _fit_arima_residuals(residuals_for_arima, config.arima_order)
        if arima_results:
            logging.info("  - Generating in-sample ARIMA predictions...")
            try:
                arima_pred_start=residuals_for_arima.index.min(); arima_pred_end=residuals_for_arima.index.max()
                if arima_pred_start > arima_pred_end: raise ValueError("ARIMA predict start > end")
                arima_residual_preds_in_sample = arima_results.predict(start=arima_pred_start,end=arima_pred_end)
                arima_residual_preds_aligned = arima_residual_preds_in_sample.reindex(oof_predictions_base.index).fillna(0.0)
                oof_predictions_final = oof_predictions_base.add(arima_residual_preds_aligned,fill_value=0.0)
                valid_indices = oof_predictions_final.dropna().index.intersection(y_base.index)
                if not valid_indices.empty:
                    final_mae = mean_absolute_error(y_base.loc[valid_indices,target_col_name_original],oof_predictions_final.loc[valid_indices])
                    logging.info(f"  - Hybrid OOF. Final Hybrid MAE: {final_mae:.4f}")
                else: logging.warning("  - No valid indices for final hybrid MAE."); final_mae = np.nan
            except Exception as e:
                logging.error(f"  - Error combining ARIMA: {e}"); traceback.print_exc(); arima_results=None; final_mae=best_mae_base; oof_predictions_final=oof_predictions_base
        else: logging.warning("  - ARIMA failed. Using base LGBM."); final_mae=best_mae_base; oof_predictions_final=oof_predictions_base

    log_step("Step 6: Calculating Feature Importance (Base LGBM)")
    feature_importance_data = []
    try:
        gain_importance=lgbm_model.feature_importances_; feature_names=lgbm_model.booster_.feature_name()
        gain_map=dict(zip(feature_names,gain_importance))
        explainer=shap.TreeExplainer(lgbm_model); shap_values=explainer.shap_values(X_base)
        if isinstance(shap_values,list): shap_values=shap_values[0]
        mean_abs_shap=np.mean(np.abs(shap_values),axis=0); shap_map=dict(zip(feature_names,mean_abs_shap))
        combined_importance=[{'feature':f,'shap_mean_abs':shap_map.get(f,0),'gain':gain_map.get(f,0)} for f in feature_names]
        combined_importance_df=pd.DataFrame(combined_importance).sort_values(by=['shap_mean_abs','gain'],ascending=False)
        top_n=config.feature_importance_top_n
        feature_importance_data=[(row['feature'],row['shap_mean_abs'],row['gain']) for i,row in combined_importance_df.head(top_n).iterrows()]
        logging.info(f"  - Calculated importance. Top SHAP: {feature_importance_data[0][0] if feature_importance_data else 'N/A'}")
    except Exception as e: logging.error(f"  - Error feature importance: {e}")

    log_step("Step 7: Creating Summary & Final Residual Diagnostics")
    summary_df=pd.DataFrame({'Actual':y_base[target_col_name_original],'CV_Prediction':oof_predictions_final})
    summary_df.index=y_base.index
    summary_df['Residual']=summary_df['Actual']-summary_df['CV_Prediction']; final_residuals=summary_df['Residual'].dropna()
    diagnostics={'dw_stat':None,'lb_pvalue':None}
    if not final_residuals.empty:
        try:
            diagnostics['dw_stat']=sm.stats.durbin_watson(final_residuals)
            lb_test=sm.stats.acorr_ljungbox(final_residuals,lags=config.ljung_box_lags,return_df=True)
            diagnostics['lb_pvalue']=lb_test['lb_pvalue'].iloc[-1]
            logging.info(f"  - Final Diags: DW={diagnostics['dw_stat']:.2f}, LB p={diagnostics['lb_pvalue']:.3f}")
        except Exception as e: logging.error(f"  - Error final diags: {e}")
    else: logging.warning("  - No final residuals for diags.")

    log_step("Step 8: Calculating Next Week Estimate")
    next_week_estimate = None; next_week_features_base = None
    last_train_date = y_base.index.max(); estimate_date = last_train_date + pd.Timedelta(weeks=1)
    logging.info(f"  - Target Date for Estimation: {estimate_date.strftime('%Y-%m-%d')}")
    try:
        estimation_features_df = pd.DataFrame(index=[estimate_date])
        start_of_week = estimate_date - pd.Timedelta(days=6)
        daily_data_for_week = all_data_vars[feature_df_name].loc[start_of_week:estimate_date].select_dtypes(include=np.number)
        if daily_data_for_week.empty: raise ValueError(f"No daily data for week {start_of_week.date()}")
        aggregated = daily_data_for_week.agg(agg_methods) 
        for col, val in aggregated.items(): estimation_features_df[sanitize_feature_name(f"Agg_{col}")] = val

        history_y=y_base[target_col_name_original]; history_diff=history_y.diff()
        sanitized_target_name = sanitize_feature_name(target_col_name_original) 
        for lag in config.target_lags:
            try: estimation_features_df[sanitize_feature_name(f'{sanitized_target_name}_lag_{lag}')] = history_y.shift(lag-1).iloc[-1]
            except IndexError: estimation_features_df[sanitize_feature_name(f'{sanitized_target_name}_lag_{lag}')] = np.nan
            try: estimation_features_df[sanitize_feature_name(f'{sanitized_target_name}_diff_lag_{lag}')] = history_diff.shift(lag-1).iloc[-1]
            except IndexError: estimation_features_df[sanitize_feature_name(f'{sanitized_target_name}_diff_lag_{lag}')] = np.nan
        diff_lag1_feat = sanitize_feature_name(f'{sanitized_target_name}_diff_lag_1')
        if diff_lag1_feat in X_base.columns:
            for window in config.rolling_windows_diff:
                estimation_features_df[sanitize_feature_name(f'{diff_lag1_feat}_roll_mean_{window}')] = history_diff.rolling(window,min_periods=1).mean().iloc[-1]
                estimation_features_df[sanitize_feature_name(f'{diff_lag1_feat}_roll_std_{window}')] = history_diff.rolling(window,min_periods=1).std().fillna(0).iloc[-1]
        lag1_feat = sanitize_feature_name(f'{sanitized_target_name}_lag_1')
        if lag1_feat in X_base.columns:
            for window in config.rolling_windows_level:
                estimation_features_df[sanitize_feature_name(f'{lag1_feat}_roll_std_{window}')] = history_y.rolling(window,min_periods=1).std().fillna(0).iloc[-1]
        estimation_features_df[sanitize_feature_name('cal_weekofyear')] = estimation_features_df.index.isocalendar().week.astype(int)
        estimation_features_df[sanitize_feature_name('cal_month')] = estimation_features_df.index.month
        estimation_features_df[sanitize_feature_name('cal_year')] = estimation_features_df.index.year
        estimation_features_df = estimation_features_df.fillna(0)
        base_model_features = lgbm_model.booster_.feature_name()
        missing_cols = set(base_model_features) - set(estimation_features_df.columns)
        if missing_cols: logging.warning(f"  Missing est cols added with 0: {missing_cols}");
        for col in missing_cols: estimation_features_df[col] = 0
        estimation_features_aligned = estimation_features_df[base_model_features]
        next_week_features_base = estimation_features_aligned.copy()
        base_estimate = lgbm_model.predict(estimation_features_aligned)[0]
        logging.info(f"  - Base LGBM Estimate: {base_estimate:.4f}")
        next_week_estimate = base_estimate
        is_hybrid_run = arima_results is not None
        if is_hybrid_run:
            logging.info("  - Forecasting ARIMA residual (1 step)...")
            try:
                arima_resid_forecast = arima_results.forecast(steps=1).iloc[0]
                logging.info(f"  - ARIMA Residual Forecast: {arima_resid_forecast:.4f}")
                next_week_estimate += arima_resid_forecast
            except Exception as e: logging.error(f"    Error forecasting ARIMA resid: {e}.")
        elif config.use_hybrid_arima and not is_hybrid_run: logging.warning("  - Hybrid intended, ARIMA failed.")
        logging.info(f"  - ** Final Estimated Value: {next_week_estimate:.4f} **")
    except Exception as e: logging.error(f"  - Error next week est: {e}"); traceback.print_exc(); next_week_estimate=None; next_week_features_base=None

    log_step(f"Pipeline Finished for {target_col_name_original}")
    return {'target':target_col_name_original,'feature_source':feature_df_name,
            'best_mae':final_mae,'pass1_mae':best_mae_base,
            'best_lgbm_params':best_params_base,'arima_results':arima_results,
            'lgbm_model':lgbm_model,'summary_df':summary_df,
            'X_base_shape':X_base.shape,'y_base_shape':y_base.shape,
            'target_std_dev':y_base[target_col_name_original].std(),
            'X_base_columns':X_base.columns.tolist(),'feature_importance':feature_importance_data,
            'diagnostics':diagnostics,'next_week_estimate':next_week_estimate,
            'next_week_features':next_week_features_base}

def run_all_pipelines(config: PipelineConfig, all_data_vars: Dict[str, pd.DataFrame],
                      target_feature_map: Dict[str, str]) -> Dict[str, Any]:
    all_results = {}
    log_step("Starting Main Execution Loop for All Targets")
    for target_orig, feat_df_name in target_feature_map.items():
        current_config = dataclasses.replace(config)
        if feat_df_name not in all_data_vars or all_data_vars[feat_df_name].empty:
            logging.error(f"Feature DF '{feat_df_name}' for '{target_orig}' missing/empty. Skipping.")
            all_results[target_orig] = None; continue
        result = _run_single_target_pipeline(target_orig, feat_df_name, all_data_vars, current_config)
        all_results[target_orig] = result
    log_step("Main Execution Loop Finished"); return all_results

def plot_feature_importance(all_results: dict, target: str, top_n: int = 15) -> Optional[plt.Figure]:
    log_step(f"Generating Feature Importance Plot for {target}")
    if target not in all_results or not all_results[target] or 'feature_importance' not in all_results[target]: logging.error(f"No importance for '{target}'."); return None
    importance_data = all_results[target]['feature_importance']
    if not importance_data: logging.warning(f"Importance empty for '{target}'."); return None
    try: features=[item[0] for item in importance_data[:top_n]]; shap_values=[item[1] for item in importance_data[:top_n]]
    except (IndexError,TypeError) as e: logging.error(f"Error extracting importance for '{target}': {e}"); return None
    fig, ax = plt.subplots(figsize=(10,max(5,top_n*0.4))); ax.barh(range(len(features)),shap_values,align='center')
    ax.set_yticks(range(len(features))); ax.set_yticklabels(features); ax.invert_yaxis()
    ax.set_xlabel("Mean(|SHAP Value|) - Avg Impact Magnitude"); ax.set_title(f"Feat Importance (Top {top_n} SHAP) for {target}"); plt.tight_layout(); return fig

def save_models(all_results: dict, path: str = './models_v3_hybrid'):
    log_step(f"Saving Models to {path}")
    os.makedirs(path,exist_ok=True); saved_lgbm=0; saved_arima=0
    for target, result in all_results.items():
        safe_target_filename = sanitize_feature_name(target)
        if result:
            if 'lgbm_model' in result and result['lgbm_model'] is not None:
                try: joblib.dump(result['lgbm_model'],os.path.join(path,f"model_lgbm_{safe_target_filename}.joblib")); logging.info(f"  Saved LGBM for '{target}'"); saved_lgbm+=1
                except Exception as e: logging.error(f"  Failed save LGBM for '{target}': {e}")
            else: logging.warning(f"  No LGBM model for '{target}'.")
            if 'arima_results' in result and result['arima_results'] is not None:
                try: joblib.dump(result['arima_results'],os.path.join(path,f"model_arima_resid_{safe_target_filename}.joblib")); logging.info(f"  Saved ARIMA for '{target}'"); saved_arima+=1
                except Exception as e: logging.error(f"  Failed save ARIMA for '{target}': {e}")
        else: logging.warning(f"  Skipping save for '{target}': No results.")
    logging.info(f"Finished saving {saved_lgbm} LGBM and {saved_arima} ARIMA models.")

# ##################################################################################
# # FUNCTION RELATED TO 4-WEEK FORECAST - COMMENTED OUT AS PER USER REQUEST
# ##################################################################################
# def forecast_weeks(target_col_name_original: str, n_weeks: int, all_results: Dict[str, Any],
#                    all_data_vars: Dict[str, pd.DataFrame], config: PipelineConfig) -> Optional[pd.DataFrame]:
#     log_step(f"Generating {n_weeks}-Week Forecast for {target_col_name_original}")
#     if target_col_name_original not in all_results or not all_results[target_col_name_original]: logging.error(f"No results for '{target_col_name_original}'."); return None
#     result = all_results[target_col_name_original]
#     lgbm_model=result.get('lgbm_model'); arima_results=result.get('arima_results')
#     feature_df_name=result.get('feature_source'); X_base_columns=result.get('X_base_columns')
#     summary_df=result.get('summary_df')
#     if not all([lgbm_model,feature_df_name,X_base_columns,summary_df is not None]): logging.error(f"Missing components for '{target_col_name_original}'."); return None
#     if feature_df_name not in all_data_vars: logging.error(f"Feature DF '{feature_df_name}' missing."); return None

#     feature_df_fcst = all_data_vars[feature_df_name] 
#     history_df = summary_df[['Actual']].rename(columns={'Actual':target_col_name_original})
#     if not isinstance(history_df.index,pd.DatetimeIndex): logging.error("History index error."); return None
#     history_df = history_df.sort_index()
#     last_hist_date = history_df.index.max()
#     forecast_dates = pd.date_range(start=last_hist_date+pd.Timedelta(weeks=1),periods=n_weeks,freq='W-FRI')
#     forecasts = pd.Series(index=forecast_dates,dtype=float,name='Forecast')
#     history_y_fcst = history_df[target_col_name_original].copy()
#     is_hybrid_run = arima_results is not None
#     sanitized_target_name = sanitize_feature_name(target_col_name_original) 

#     agg_methods = {}
#     numeric_cols_fcst = feature_df_fcst.select_dtypes(include=np.number).columns
#     for col in numeric_cols_fcst:
#         col_lower=str(col).lower();
#         if any(k in col_lower for k in ['price','temp','avg','pct','share','hdd','cdd']): agg_methods[col]='mean'
#         elif any(k in col_lower for k in ['storage','level','inventory']): agg_methods[col]='last'
#         else: agg_methods[col]='sum'

#     for i, current_forecast_date in enumerate(forecast_dates):
#         logging.info(f"  - Forecasting week {i+1}/{n_weeks}: {current_forecast_date.strftime('%Y-%m-%d')}")
#         estimation_features_df = pd.DataFrame(index=[current_forecast_date])
#         start_of_week = current_forecast_date - pd.Timedelta(days=6)
#         try:
#             daily_data_for_week = feature_df_fcst.loc[start_of_week:current_forecast_date, numeric_cols_fcst]
#             if daily_data_for_week.empty: raise ValueError(f"No daily numeric data for week {start_of_week.date()}")
#             aggregated = daily_data_for_week.agg(agg_methods)
#             for col, val in aggregated.items(): estimation_features_df[sanitize_feature_name(f"Agg_{col}")] = val
#         except Exception as e: logging.error(f"ERR aggregating daily for forecast {current_forecast_date}: {e}"); return None
#         history_diff = history_y_fcst.diff()
#         for lag in config.target_lags:
#             try: estimation_features_df[sanitize_feature_name(f'{sanitized_target_name}_lag_{lag}')] = history_y_fcst.shift(lag).iloc[-1]
#             except IndexError: estimation_features_df[sanitize_feature_name(f'{sanitized_target_name}_lag_{lag}')] = np.nan
#             try: estimation_features_df[sanitize_feature_name(f'{sanitized_target_name}_diff_lag_{lag}')] = history_diff.shift(lag).iloc[-1]
#             except IndexError: estimation_features_df[sanitize_feature_name(f'{sanitized_target_name}_diff_lag_{lag}')] = np.nan
#         diff_lag1_feat = sanitize_feature_name(f'{sanitized_target_name}_diff_lag_1')
#         if diff_lag1_feat in X_base_columns:
#             for window in config.rolling_windows_diff:
#                 estimation_features_df[sanitize_feature_name(f'{diff_lag1_feat}_roll_mean_{window}')] = history_diff.rolling(window,min_periods=1).mean().iloc[-1]
#                 estimation_features_df[sanitize_feature_name(f'{diff_lag1_feat}_roll_std_{window}')] = history_diff.rolling(window,min_periods=1).std().fillna(0).iloc[-1]
#         lag1_feat = sanitize_feature_name(f'{sanitized_target_name}_lag_1')
#         if lag1_feat in X_base_columns:
#             for window in config.rolling_windows_level:
#                 estimation_features_df[sanitize_feature_name(f'{lag1_feat}_roll_std_{window}')] = history_y_fcst.rolling(window,min_periods=1).std().fillna(0).iloc[-1]
#         estimation_features_df[sanitize_feature_name('cal_weekofyear')] = estimation_features_df.index.isocalendar().week.astype(int)
#         estimation_features_df[sanitize_feature_name('cal_month')] = estimation_features_df.index.month
#         estimation_features_df[sanitize_feature_name('cal_year')] = estimation_features_df.index.year
#         estimation_features_df = estimation_features_df.fillna(0)
#         missing_cols = set(X_base_columns) - set(estimation_features_df.columns)
#         if missing_cols: logging.warning(f"  Forecast {i+1}: Missing cols: {missing_cols}");
#         for col in missing_cols: estimation_features_df[col] = 0
#         try: estimation_features_aligned = estimation_features_df[X_base_columns]
#         except Exception as e: logging.error(f"ERR aligning forecast features for {current_forecast_date}: {e}"); return None
#         try: base_prediction = lgbm_model.predict(estimation_features_aligned)[0]
#         except Exception as e: logging.error(f"Error predicting BASE for {current_forecast_date}: {e}"); return None
#         arima_resid_prediction = 0.0
#         if is_hybrid_run:
#             try: arima_resid_prediction = arima_results.forecast(steps=i+1).iloc[-1]
#             except Exception as e: logging.warning(f"Error ARIMA resid step {i+1} for {current_forecast_date}: {e}. Using 0.")
#         final_prediction = base_prediction + arima_resid_prediction
#         forecasts.loc[current_forecast_date] = final_prediction
#         history_y_fcst.loc[current_forecast_date] = final_prediction
#         logging.info(f"    -> Forecast: {final_prediction:.4f} (Base: {base_prediction:.4f}, Resid: {arima_resid_prediction:.4f})")
#     log_step(f"Finished {n_weeks}-Week Forecast for {target_col_name_original}"); return forecasts.to_frame()
# ##################################################################################

# === Main Execution Flow ===
if __name__ == '__main__': 
    log_step("Main Execution: Running Pipelines")
    pipeline_config = PipelineConfig(optuna_trials=40, use_hybrid_arima=True, arima_order=(2,0,1)) 
    logging.info(f"Pipeline Config: {pipeline_config}")

    if 'dataframe_vars' not in globals() or not dataframe_vars:
        logging.error("CRITICAL: `dataframe_vars` was not created or is empty. This likely means data preparation steps failed.")
        logging.error("Please check logs from 'Defining and Verifying Required Variables for Pipeline' and previous data loading/processing steps.")
    else:
        all_results = run_all_pipelines(pipeline_config, dataframe_vars, TARGET_FEATURE_MAP)

# ##################################################################################
# # SECTION RELATED TO CALLING 4-WEEK FORECAST - COMMENTED OUT AS PER USER REQUEST
# ##################################################################################
#         log_step("Main Execution: Generating Forecasts")
#         forecast_target = 'Total Lower 48'; n_forecast_weeks = 4
#         if forecast_target in all_results and all_results[forecast_target]:
#             last_actual_date = all_results[forecast_target]['summary_df'].index.max()
#             required_end_date = last_actual_date + pd.Timedelta(weeks=n_forecast_weeks)
#             feature_df_for_forecast = dataframe_vars[TARGET_FEATURE_MAP[forecast_target]]
#             if feature_df_for_forecast.index.max() >= required_end_date:
#                 logging.info(f"Attempting {n_forecast_weeks}-week forecast for {forecast_target}...")
#                 forecast_df_total = forecast_weeks(forecast_target, n_forecast_weeks, all_results, dataframe_vars, pipeline_config)
#                 if forecast_df_total is not None:
#                     all_results[forecast_target]['fw4'] = forecast_df_total
#                     logging.info(f"\nGenerated {n_forecast_weeks}-Week Forecast for {forecast_target}:\n{forecast_df_total}")
#                 else:
#                     logging.error(f"Multi-week forecast failed for {forecast_target}.")
#                     if forecast_target in all_results and all_results[forecast_target]: all_results[forecast_target]['fw4'] = None
#             else:
#                 logging.warning(f"Skipping multi-week forecast for '{forecast_target}': Insufficient future daily data. Max date: {feature_df_for_forecast.index.max()}, Required > {required_end_date}")
#                 if forecast_target in all_results and all_results[forecast_target]: all_results[forecast_target]['fw4'] = None
#         else:
#             logging.warning(f"Skipping multi-week forecast: No results found for target '{forecast_target}'.")
# ##################################################################################

        log_step("Self-Check Summary")
        print(f"{'Target':<25} {'Final MAE':<10} {'Base MAE':<10} {'Last Actual':<12} {'Last Pred':<12} {'Next Est':<12} {'StdDev':<10} {'DW Stat':<10} {'LB p-val':<10}")
        print("-" * 130)
        # Guard against all_results not being defined if the initial check for dataframe_vars fails
        if 'all_results' in locals() and all_results: 
            for target, res in all_results.items():
                if res:
                    mae = f"{res.get('best_mae',np.nan):.2f}"; p1_mae = f"{res.get('pass1_mae',np.nan):.2f}"
                    last_actual_val,last_pred_val = np.nan,np.nan
                    if 'summary_df' in res and not res['summary_df'].empty:
                        if 'Actual' in res['summary_df'].columns and not res['summary_df']['Actual'].dropna().empty: last_actual_val = res['summary_df']['Actual'].dropna().iloc[-1]
                        if 'CV_Prediction' in res['summary_df'].columns and not res['summary_df']['CV_Prediction'].dropna().empty: last_pred_val = res['summary_df']['CV_Prediction'].dropna().iloc[-1]
                    last_actual=f"{last_actual_val:.2f}" if pd.notna(last_actual_val) else "N/A"
                    last_pred=f"{last_pred_val:.2f}" if pd.notna(last_pred_val) else "N/A"
                    next_est_val = res.get('next_week_estimate'); next_est = f"{next_est_val:.2f}" if pd.notna(next_est_val) else "N/A"
                    stddev=f"{res.get('target_std_dev',np.nan):.2f}"
                    dw=f"{res['diagnostics'].get('dw_stat',np.nan):.2f}" if 'diagnostics' in res else "N/A"
                    lbp=f"{res['diagnostics'].get('lb_pvalue',np.nan):.3f}" if 'diagnostics' in res else "N/A"
                else: mae,p1_mae,last_actual,last_pred,next_est,stddev,dw,lbp = ("Failed",)*8
                print(f"{target:<25} {mae:<10} {p1_mae:<10} {last_actual:<12} {last_pred:<12} {next_est:<12} {stddev:<10} {dw:<10} {lbp:<10}")
        else:
            print("Skipping summary table as 'all_results' is not available or pipelines did not run.")


        # Guard against all_results not being defined
        if 'all_results' in locals() and all_results:
            log_step("Main Execution: Saving Models")
            save_models(all_results, path='./models_v3_hybrid') 

            log_step("Main Execution: Displaying Feature Importance")
            target_to_plot = 'Total Lower 48'
            if target_to_plot in all_results and all_results[target_to_plot]:
                fig = plot_feature_importance(all_results, target_to_plot)
                if fig:
                    plt.show()
                else: logging.warning(f"Failed to generate importance plot for {target_to_plot}.")
            else: logging.warning(f"Cannot plot importance for '{target_to_plot}': Check pipeline results.")
        else:
            logging.warning("Skipping model saving and feature importance plotting as pipeline results are not available.")

        log_step("Execution Complete")

# === Cell 7: Deployment Comments (Informational) ===
# (These are comments from the original notebook, kept for your reference)
# ... (rest of the comments are kept as is) ...