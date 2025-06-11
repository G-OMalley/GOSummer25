import os
import requests
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
import time
import calendar 
from dateutil.relativedelta import relativedelta 

# --- Global Configurations ---
TARGET_SERIES = {
    "N9070US2": "Prod",
    "N9102CN2": "CadIMP",
    "N3045US2": "Power Burn",
    "N3035US2": "Industrial",
    "N3010US2": "_Temp_ResidentialCons",
    "N3020US2": "_Temp_CommercialCons",
    "N9132MX2": "MexExp",
    "N9133US2": "LNGExp"
}

FINAL_CSV_COLUMNS = [
    "Prod", "CadIMP", "Power Burn", "Industrial", "ResCom", "MexExp", "LNGExp"
]

API_BASE_PATH_TEMPLATE = "https://api.eia.gov/v2/{category_path}/data/" 

SERIES_CATEGORY_PATHS = {
    "N9070US2": "natural-gas/prod/sum",      
    "N9102CN2": "natural-gas/move/impc",     
    "N3045US2": "natural-gas/cons/sum",      
    "N3035US2": "natural-gas/cons/sum",      
    "N3010US2": "natural-gas/cons/sum",      
    "N3020US2": "natural-gas/cons/sum",      
    "N9132MX2": "natural-gas/move/expc",     
    "N9133US2": "natural-gas/move/expc"      
}

DEFAULT_START_PERIOD = "2015-01" 
REFRESH_MONTHS = 12 

SCRIPT_DIR_GLOBAL = Path(__file__).resolve().parent 
PROJECT_ROOT_DIR_GLOBAL = SCRIPT_DIR_GLOBAL.parent 
OUTPUT_DIR_GLOBAL = PROJECT_ROOT_DIR_GLOBAL / "INFO"
OUTPUT_FILENAME_GLOBAL = "EIAFundamentalMonthlydayAvg.csv" 
API_REQUEST_DELAY_SECONDS_GLOBAL = 1 

# --- Helper Functions ---
def load_api_key():
    """Loads the EIA API key from .env file."""
    current_script_dir = Path(__file__).resolve().parent
    dotenv_path_script_level = current_script_dir / '.env' 
    dotenv_path_project_level = current_script_dir.parent / '.env'
    
    loaded_path = None
    if dotenv_path_script_level.exists():
        load_dotenv(dotenv_path=dotenv_path_script_level)
        loaded_path = dotenv_path_script_level
    elif dotenv_path_project_level.exists():
        load_dotenv(dotenv_path=dotenv_path_project_level)
        loaded_path = dotenv_path_project_level
    else:
        print(f"Warning: .env file not found at {dotenv_path_script_level} or {dotenv_path_project_level}.")
        print("Attempting to load from default locations (e.g., current working directory).")
        if not load_dotenv(): 
             print("Warning: No .env file loaded by default method either.")
    
    if loaded_path:
        print(f"Loaded .env from {loaded_path}")
    elif os.getenv("EIA_API_KEY"): 
        print("Loaded EIA_API_KEY from active environment variables.")
    
    api_key_raw = os.getenv("EIA_API_KEY")
    if not api_key_raw:
        print("Error: EIA_API_KEY not found in environment variables.")
        return None
    return api_key_raw.strip()

def fetch_series_data(api_key, series_id, temp_series_name, category_path, start_period_for_fetch):
    """
    Fetches data for a single series from start_period_for_fetch, calculates daily average, 
    and returns a pandas DataFrame. Assumes original data is monthly MMcf.
    """
    print(f"Fetching data for: {temp_series_name} (ID: {series_id}) from {start_period_for_fetch} using path: {category_path}...")
    
    base_url = API_BASE_PATH_TEMPLATE.format(category_path=category_path) 
    params = {
        "api_key": api_key,
        "frequency": "monthly",
        "data[0]": "value",
        "facets[series][0]": series_id, 
        "start": start_period_for_fetch,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc", 
        "offset": "0",
        "length": "5000" 
    }
    response_obj = None 
    try:
        response_obj = requests.get(base_url, params=params, timeout=(10, 30))
        response_obj.raise_for_status()
        api_data = response_obj.json()
    except requests.exceptions.Timeout:
        print(f"Error: Timeout while fetching {temp_series_name}.")
        return None
    except requests.exceptions.HTTPError as http_err:
        print(f"Error: HTTP error for {temp_series_name} (ID: {series_id}) with URL {response_obj.url if response_obj else base_url}: {http_err}")
        if response_obj is not None: print(f"Response content: {response_obj.text[:300]}...")
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"Error: Request exception for {temp_series_name}: {req_err}")
        return None
    except requests.exceptions.JSONDecodeError:
        print(f"Error: Could not decode JSON for {temp_series_name}.")
        if response_obj is not None: print(f"Response content: {response_obj.text[:300]}...")
        return None

    if 'response' not in api_data or 'data' not in api_data['response']:
        print(f"Error: Unexpected API response format for {temp_series_name}. Missing 'response' or 'data' key.")
        return None

    data_list = api_data['response']['data']
    total_api_records_str = api_data['response'].get('total', '0')
    try: total_api_records = int(total_api_records_str)
    except ValueError: total_api_records = 0

    if not data_list and total_api_records == 0 :
        print(f"No new data returned by API for {temp_series_name} (ID: {series_id}) from {start_period_for_fetch} with path {category_path}.")
        empty_df = pd.DataFrame(columns=[temp_series_name]); empty_df.index.name = 'period'     
        return empty_df

    processed_data = []
    for entry in data_list:
        period_str = entry.get('period')
        monthly_value_str = entry.get('value') 
        if entry.get('series') and entry.get('series') != series_id: continue 
        if period_str and monthly_value_str is not None:
            try:
                monthly_value_mmcf = float(monthly_value_str)
                year = int(period_str.split('-')[0]); month = int(period_str.split('-')[1])
                _, days_in_month = calendar.monthrange(year, month)
                if days_in_month <= 0: 
                    print(f"Warning: Invalid days_in_month ({days_in_month}) for {period_str} in series {temp_series_name}. Skipping.")
                    continue
                daily_average_bcf = (monthly_value_mmcf / days_in_month) / 1000
                processed_data.append({'period': period_str, temp_series_name: daily_average_bcf})
            except (ValueError, TypeError, IndexError) as calc_e:
                print(f"Warning: Could not process/calculate value for '{monthly_value_str}' for {temp_series_name} at {period_str}. Error: {calc_e}")
        elif period_str: processed_data.append({'period': period_str, temp_series_name: None})

    if not processed_data:
        empty_df = pd.DataFrame(columns=[temp_series_name]); empty_df.index.name = 'period'
        return empty_df
    df = pd.DataFrame(processed_data)
    if df.empty: 
        empty_df = pd.DataFrame(columns=[temp_series_name]); empty_df.index.name = 'period'
        return empty_df
    df['period'] = pd.to_datetime(df['period'], format='%Y-%m')
    df = df.set_index('period')
    return df

# --- Main Script ---
def main():
    api_key = load_api_key()
    if not api_key:
        return

    output_file_path = OUTPUT_DIR_GLOBAL / OUTPUT_FILENAME_GLOBAL 
    df_existing = None
    start_period_for_fetch = DEFAULT_START_PERIOD 
    refresh_start_date_dt = None 

    if output_file_path.exists():
        try:
            df_existing = pd.read_csv(output_file_path, index_col="Period") 
            if not df_existing.empty:
                df_existing.index = pd.to_datetime(df_existing.index, errors='coerce') 
                df_existing.dropna(axis=0, how='all', subset=df_existing.columns, inplace=True) 
                df_existing = df_existing[pd.notna(df_existing.index)]

                if not df_existing.index.empty and isinstance(df_existing.index, pd.DatetimeIndex):
                    latest_date_in_csv = df_existing.index.max()
                    if pd.notna(latest_date_in_csv):
                        refresh_start_date_dt = latest_date_in_csv - relativedelta(months=REFRESH_MONTHS-1) 
                        refresh_start_date_dt = refresh_start_date_dt.replace(day=1) 
                        
                        start_period_for_fetch = refresh_start_date_dt.strftime('%Y-%m')
                        print(f"Existing CSV found. Latest period: {latest_date_in_csv.strftime('%Y-%m')}.")
                        print(f"Refreshing data from: {start_period_for_fetch} (approx. last {REFRESH_MONTHS} months + new data).")
                        
                        df_to_keep = df_existing[df_existing.index < refresh_start_date_dt]
                        if df_to_keep.empty and not df_existing.empty:
                             print(f"All existing data falls within the {REFRESH_MONTHS}-month refresh window and will be replaced/re-fetched.")
                        elif not df_existing.empty:
                             print(f"Keeping existing data before {refresh_start_date_dt.strftime('%Y-%m')}.")
                        df_existing = df_to_keep 
                    else:
                        print(f"Warning: Could not determine latest date from 'Period' column in {output_file_path}. Using default start: {DEFAULT_START_PERIOD}")
                        df_existing = None 
                else:
                     print(f"Existing CSV {output_file_path} 'Period' index issue or empty after NaT drop. Using default start: {DEFAULT_START_PERIOD}")
                     df_existing = None 
            else:
                print(f"Existing CSV {output_file_path} is empty. Using default start: {DEFAULT_START_PERIOD}")
                df_existing = None 
        except Exception as e:
            print(f"Error reading or processing existing CSV {output_file_path}: {e}. Using default start: {DEFAULT_START_PERIOD}")
            df_existing = None
    else:
        print(f"No existing CSV found at {output_file_path}. Starting fresh from: {DEFAULT_START_PERIOD}")

    all_new_series_dfs = []
    any_new_data_fetched_for_any_series = False

    for series_id, temp_col_name in TARGET_SERIES.items(): 
        category_path = SERIES_CATEGORY_PATHS.get(series_id) 
        if not category_path:
            df = pd.DataFrame(columns=[temp_col_name]); df.index.name = 'period'
            all_new_series_dfs.append(df); continue

        df_new = fetch_series_data(api_key, series_id, temp_col_name, category_path, start_period_for_fetch) 
        if df_new is not None: 
            if not df_new.empty: any_new_data_fetched_for_any_series = True
            all_new_series_dfs.append(df_new)
        else: 
            df = pd.DataFrame(columns=[temp_col_name]); df.index.name = 'period'
            all_new_series_dfs.append(df)
        
        print(f"Waiting {API_REQUEST_DELAY_SECONDS_GLOBAL} second(s) before next request...\n")
        time.sleep(API_REQUEST_DELAY_SECONDS_GLOBAL)


    if not any_new_data_fetched_for_any_series:
        print(f"No new data fetched for any series for the period starting from {start_period_for_fetch}.")
        if df_existing is not None and not df_existing.empty:
             print(f"Saving existing data (before {refresh_start_date_dt.strftime('%Y-%m') if refresh_start_date_dt else 'refresh window'}) back to {output_file_path}.")
             if isinstance(df_existing.index, pd.DatetimeIndex): 
                df_existing.index = df_existing.index.strftime('%Y-%m')
             df_existing.index.name = "Period"
             # Ensure only final columns are saved if df_existing was from an old run
             cols_to_save = [col for col in FINAL_CSV_COLUMNS if col in df_existing.columns]
             if cols_to_save:
                df_existing[cols_to_save].to_csv(output_file_path, float_format='%.4f')
             else:
                print("Warning: Existing data frame has no columns matching FINAL_CSV_COLUMNS. Not saving.")
        elif df_existing is None : 
             print("No existing data and no new data. No CSV file written or modified.")
        return
    
    print("New data fetched. Combining with existing data (if any)...")

    all_new_series_dfs_filtered = [df for df in all_new_series_dfs if df is not None]
    if not all_new_series_dfs_filtered: # Should be caught by any_new_data_fetched_for_any_series
        print("No valid new data frames to concatenate. This should not happen if any_new_data_fetched_for_any_series was true.")
        return

    newly_fetched_combined_df = pd.concat(all_new_series_dfs_filtered, axis=1, join='outer')

    if df_existing is not None and not df_existing.empty:
        if not isinstance(df_existing.index, pd.DatetimeIndex):
            df_existing.index = pd.to_datetime(df_existing.index, errors='coerce')
            df_existing = df_existing[pd.notna(df_existing.index)] 
        if not isinstance(newly_fetched_combined_df.index, pd.DatetimeIndex):
            newly_fetched_combined_df.index = pd.to_datetime(newly_fetched_combined_df.index, errors='coerce')
            newly_fetched_combined_df = newly_fetched_combined_df[pd.notna(newly_fetched_combined_df.index)]
        
        # Important: Ensure ResCom from df_existing is preserved if it's there, before it gets recalculated
        # for the newly fetched part.
        # We will handle ResCom calculation after fully combining Prod, CadIMP, temps, etc.
        # The df_existing already has its final columns (including a potentially correct old ResCom).
        # The newly_fetched_combined_df has Prod, CadIMP, _Temp_Res, _Temp_Com etc.
        
        # Strategy: Combine all base columns first. Then calculate ResCom on the whole thing.
        # This means df_existing should NOT have ResCom if we are recalculating it based on its components.
        # Or, we preserve existing ResCom and only calculate for new rows. The current logic handles this.
        
        final_df = pd.concat([df_existing, newly_fetched_combined_df])

    else:
        final_df = newly_fetched_combined_df
        
    if not final_df.index.is_unique:
        final_df = final_df[~final_df.index.duplicated(keep='last')]
    
    final_df = final_df.sort_index()

    # --- Perform Summing and Column Finalization ---
    res_col_name = "_Temp_ResidentialCons"
    com_col_name = "_Temp_CommercialCons"
    
    # Initialize "ResCom" if it doesn't exist (e.g. first run)
    # If it exists from df_existing, its values for older periods will be preserved.
    if "ResCom" not in final_df.columns:
        final_df["ResCom"] = pd.NA # Use NA for missing numeric, or 0.0 if you prefer

    # Identify rows that have new _Temp_ data (i.e., rows from newly_fetched_combined_df)
    # These are the rows where ResCom should be (re)calculated.
    mask_new_res_data = final_df[res_col_name].notna() if res_col_name in final_df.columns else pd.Series(False, index=final_df.index)
    mask_new_com_data = final_df[com_col_name].notna() if com_col_name in final_df.columns else pd.Series(False, index=final_df.index)
    rows_to_recalculate_rescom = mask_new_res_data | mask_new_com_data

    # Calculate ResCom for these specific rows
    val_res = final_df.loc[rows_to_recalculate_rescom, res_col_name].fillna(0) if res_col_name in final_df.columns else 0
    val_com = final_df.loc[rows_to_recalculate_rescom, com_col_name].fillna(0) if com_col_name in final_df.columns else 0
    
    final_df.loc[rows_to_recalculate_rescom, "ResCom"] = val_res + val_com
    
    # If for some reason ResCom was not initialized and no temp data, ensure column exists
    if "ResCom" not in final_df.columns:
         final_df["ResCom"] = pd.NA


    cols_to_drop = [col for col in [res_col_name, com_col_name] if col in final_df.columns]
    if cols_to_drop: final_df.drop(columns=cols_to_drop, inplace=True)
    
    # Ensure all desired final columns exist and reorder
    current_cols = list(final_df.columns)
    final_cols_ordered = []
    for col in FINAL_CSV_COLUMNS: 
        if col in current_cols: 
            final_cols_ordered.append(col)
        else: # This case should be rare if ResCom is handled above
            final_df[col] = pd.NA 
            final_cols_ordered.append(col)
            print(f"Note: Column '{col}' was not present after processing, added as empty.")
            
    final_df = final_df[final_cols_ordered] 

    if isinstance(final_df.index, pd.DatetimeIndex):
        final_df.index = final_df.index.strftime('%Y-%m')
    final_df.index.name = "Period"

    OUTPUT_DIR_GLOBAL.mkdir(parents=True, exist_ok=True) 

    try:
        final_df.to_csv(output_file_path, float_format='%.4f')
        print(f"\nSuccessfully saved data to: {output_file_path}")
        if not final_df.empty:
            print("Columns in CSV:", list(final_df.columns))
            print("First few rows of data:"); print(final_df.head())
            print("Last few rows of data:"); print(final_df.tail())
        else: print("The final DataFrame was empty.")
    except Exception as e: print(f"Error saving data to CSV: {e}")

if __name__ == "__main__":
    main()