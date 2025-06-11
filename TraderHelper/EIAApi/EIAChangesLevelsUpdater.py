import os
import requests
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
import time
from dateutil.relativedelta import relativedelta 

# --- Global Configurations ---
WEEKLY_STORAGE_SERIES_TO_FETCH = {
    "NW2_EPG0_SWO_R48_BCF": "Lower 48 States Storage (Bcf)",
    "NW2_EPG0_SWO_R31_BCF": "East Region Storage (Bcf)",
    "NW2_EPG0_SWO_R32_BCF": "Midwest Region Storage (Bcf)",
    "NW2_EPG0_SSO_R33_BCF": "Salt Region SC Storage (Bcf)",
    "NW2_EPG0_SNO_R33_BCF": "Nonsalt Region SC Storage (Bcf)",
    "NW2_EPG0_SWO_R33_BCF": "South Central Region Storage (Bcf)",
    "NW2_EPG0_SWO_R34_BCF": "Mountain Region Storage (Bcf)",
    "NW2_EPG0_SWO_R35_BCF": "Pacific Region Storage (Bcf)"
}

API_BASE_PATH_TEMPLATE = "https://api.eia.gov/v2/{category_path}/data/" 
STORAGE_CATEGORY_PATH = "natural-gas/stor/wkly" 

DEFAULT_START_PERIOD = "2010-01-01" 
REFRESH_WEEKS = 15 

SCRIPT_DIR_GLOBAL = Path(__file__).resolve().parent 
PROJECT_ROOT_DIR_GLOBAL = SCRIPT_DIR_GLOBAL.parent 
OUTPUT_DIR_GLOBAL = PROJECT_ROOT_DIR_GLOBAL / "INFO"
TOTALS_FILENAME_GLOBAL = "EIAtotals.csv" 
CHANGES_FILENAME_GLOBAL = "EIAchanges.csv" 
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
        print("Attempting to load from default locations.")
        if not load_dotenv(): 
             print("Warning: No .env file loaded by default method either.")
    
    if loaded_path: print(f"Loaded .env from {loaded_path}")
    elif os.getenv("EIA_API_KEY"): print("Loaded EIA_API_KEY from active environment variables.")
    
    api_key_raw = os.getenv("EIA_API_KEY")
    if not api_key_raw:
        print("Error: EIA_API_KEY not found in environment variables.")
        return None
    return api_key_raw.strip()

def fetch_individual_series_data(api_key, category_path, series_id_to_fetch, start_period_for_fetch, column_name_for_df):
    """
    Fetches weekly data for a specific series_id from start_period_for_fetch.
    Returns a pandas DataFrame.
    """
    print(f"Fetching weekly data for Series ID: {series_id_to_fetch} ({column_name_for_df}) from {start_period_for_fetch}...")
    
    base_url = API_BASE_PATH_TEMPLATE.format(category_path=category_path) 
    params = {
        "api_key": api_key,
        "frequency": "weekly", 
        "data[0]": "value",    
        "facets[series][0]": series_id_to_fetch, 
        "start": start_period_for_fetch, 
        "sort[0][column]": "period",
        "sort[0][direction]": "asc", 
        "offset": "0",
        "length": "5000" 
    }

    # For debugging: print the exact URL being requested
    try:
        prepared_request = requests.Request('GET', base_url, params=params)
        prepared_url = prepared_request.prepare().url
        print(f"Attempting to fetch: {prepared_url}")
    except Exception as e_prep:
        print(f"Error preparing URL for debugging: {e_prep}")
        # Fallback to simple print if preparation fails, though less accurate for actual request encoding
        print(f"Base URL for fetch: {base_url} with params: {params}")


    response_obj = None 
    try:
        response_obj = requests.get(base_url, params=params, timeout=(10, 45)) 
        response_obj.raise_for_status()
        api_data = response_obj.json()
    except requests.exceptions.Timeout:
        print(f"Error: Timeout while fetching weekly data for series {series_id_to_fetch}.")
        return None
    except requests.exceptions.HTTPError as http_err:
        # The URL is already part of the http_err object if response_obj is set by requests
        print(f"Error: HTTP error for weekly data series {series_id_to_fetch}: {http_err}") 
        if response_obj is not None: print(f"Response content: {response_obj.text[:300]}...")
        return None
    except Exception as e: 
        print(f"Error: Exception for weekly data series {series_id_to_fetch}: {e}")
        if response_obj is not None: print(f"Response content: {response_obj.text[:300]}...")
        return None

    if 'response' not in api_data or 'data' not in api_data['response']:
        print(f"Error: Unexpected API response format for weekly data series {series_id_to_fetch}.")
        if 'error' in api_data: print(f"API Error: {api_data['error']}")
        return None

    data_list = api_data['response']['data']
    total_api_records_str = api_data['response'].get('total', '0')
    try: total_api_records = int(total_api_records_str)
    except ValueError: total_api_records = 0

    if not data_list and total_api_records == 0 :
        print(f"No new data returned by API for series {series_id_to_fetch} from {start_period_for_fetch}.")
        empty_df = pd.DataFrame(columns=[column_name_for_df]); empty_df.index.name = 'period'     
        return empty_df

    processed_data = []
    for entry in data_list:
        period_str = entry.get('period') 
        value_str = entry.get('value') 
        
        if entry.get('series') and entry.get('series') != series_id_to_fetch:
            print(f"Warning: API returned data for series {entry.get('series')} when {series_id_to_fetch} was requested for column {column_name_for_df}. Skipping.")
            continue 

        if period_str and value_str is not None:
            try:
                value = float(value_str)
                processed_data.append({'period': period_str, column_name_for_df: value})
            except (ValueError, TypeError) as val_e:
                print(f"Warning: Could not convert value '{value_str}' for {column_name_for_df} at {period_str}. Error: {val_e}")
        elif period_str: 
             processed_data.append({'period': period_str, column_name_for_df: None})

    if not processed_data:
        empty_df = pd.DataFrame(columns=[column_name_for_df]); empty_df.index.name = 'period'
        return empty_df
        
    df = pd.DataFrame(processed_data)
    if df.empty: 
        empty_df = pd.DataFrame(columns=[column_name_for_df]); empty_df.index.name = 'period'
        return empty_df
        
    df['period'] = pd.to_datetime(df['period'], format='%Y-%m-%d', errors='coerce') 
    df = df.dropna(subset=['period']) 
    df = df.set_index('period')
    return df

# --- Main Script ---
def main():
    api_key = load_api_key()
    if not api_key:
        return

    totals_output_file_path = OUTPUT_DIR_GLOBAL / TOTALS_FILENAME_GLOBAL
    df_existing_totals = None
    start_period_for_fetch = DEFAULT_START_PERIOD
    refresh_start_date_dt = None 

    if totals_output_file_path.exists():
        try:
            df_existing_totals = pd.read_csv(totals_output_file_path, index_col="Period") 
            if not df_existing_totals.empty:
                df_existing_totals.index = pd.to_datetime(df_existing_totals.index, errors='coerce') 
                df_existing_totals = df_existing_totals[pd.notna(df_existing_totals.index)]

                if not df_existing_totals.index.empty and isinstance(df_existing_totals.index, pd.DatetimeIndex):
                    latest_date_in_csv = df_existing_totals.index.max()
                    if pd.notna(latest_date_in_csv):
                        refresh_start_date_dt = latest_date_in_csv - relativedelta(weeks=REFRESH_WEEKS-1) 
                        start_period_for_fetch = refresh_start_date_dt.strftime('%Y-%m-%d')
                        
                        print(f"Existing Totals CSV found. Latest period: {latest_date_in_csv.strftime('%Y-%m-%d')}.")
                        print(f"Refreshing Totals data from: {start_period_for_fetch}.")
                        
                        df_to_keep = df_existing_totals[df_existing_totals.index < refresh_start_date_dt]
                        if df_to_keep.empty and not df_existing_totals.empty:
                             print(f"All existing Totals data falls within the {REFRESH_WEEKS}-week refresh window.")
                        elif not df_existing_totals.empty:
                             print(f"Keeping existing Totals data before {refresh_start_date_dt.strftime('%Y-%m-%d')}.")
                        df_existing_totals = df_to_keep 
                    else: 
                        df_existing_totals = None 
                else: 
                     df_existing_totals = None 
            else: 
                df_existing_totals = None 
        except Exception as e:
            print(f"Error reading existing Totals CSV {totals_output_file_path}: {e}.")
            df_existing_totals = None
    else:
        print(f"No existing Totals CSV found at {totals_output_file_path}. Starting fresh from {DEFAULT_START_PERIOD}.")

    all_new_storage_dfs = []
    any_new_data_fetched = False
    for series_id, column_name in WEEKLY_STORAGE_SERIES_TO_FETCH.items():
        df_new = fetch_individual_series_data(
            api_key, 
            STORAGE_CATEGORY_PATH, 
            series_id, 
            start_period_for_fetch, 
            column_name 
        )
        if df_new is not None:
            if not df_new.empty:
                any_new_data_fetched = True
            all_new_storage_dfs.append(df_new)
        else: 
            df_placeholder = pd.DataFrame(columns=[column_name])
            df_placeholder.index.name = 'period'
            all_new_storage_dfs.append(df_placeholder)
        
        print(f"Waiting {API_REQUEST_DELAY_SECONDS_GLOBAL} second(s)...\n")
        time.sleep(API_REQUEST_DELAY_SECONDS_GLOBAL)
        
    final_totals_df = None
    if any_new_data_fetched:
        print("New storage data fetched. Combining...")
        valid_new_dfs = [df for df in all_new_storage_dfs if df is not None and not df.columns.empty]

        if valid_new_dfs:
            newly_fetched_combined_df = pd.concat(valid_new_dfs, axis=1, join='outer')

            if df_existing_totals is not None and not df_existing_totals.empty:
                final_totals_df = pd.concat([df_existing_totals, newly_fetched_combined_df])
            else:
                final_totals_df = newly_fetched_combined_df
            
            if not final_totals_df.index.is_unique:
                final_totals_df = final_totals_df[~final_totals_df.index.duplicated(keep='last')]
            final_totals_df = final_totals_df.sort_index()
        elif df_existing_totals is not None and not df_existing_totals.empty: 
             final_totals_df = df_existing_totals.sort_index()
             print("No new valid series data was fetched, using existing totals.")
        else:
            print("No new valid series data and no existing data.")

    elif df_existing_totals is not None and not df_existing_totals.empty:
        print("No new storage data fetched. Using existing data for EIAtotals_storage_weekly.csv.")
        final_totals_df = df_existing_totals.sort_index()
    else:
        print("No existing storage data and no new storage data fetched. EIAtotals_storage_weekly.csv will not be created/updated.")


    if final_totals_df is not None and not final_totals_df.empty:
        OUTPUT_DIR_GLOBAL.mkdir(parents=True, exist_ok=True)
        df_to_save_totals = final_totals_df.copy()
        
        final_totals_columns_ordered = []
        for series_id_key, col_name_val in WEEKLY_STORAGE_SERIES_TO_FETCH.items():
            if col_name_val not in df_to_save_totals.columns:
                df_to_save_totals[col_name_val] = pd.NA 
            final_totals_columns_ordered.append(col_name_val)
        df_to_save_totals = df_to_save_totals[final_totals_columns_ordered]


        if isinstance(df_to_save_totals.index, pd.DatetimeIndex):
            df_to_save_totals.index = df_to_save_totals.index.strftime('%Y-%m-%d')
        df_to_save_totals.index.name = "Period"
        try:
            df_to_save_totals.to_csv(totals_output_file_path, float_format='%.0f') 
            print(f"\nSuccessfully saved/updated data to: {totals_output_file_path}")
            print("First few rows of EIAtotals_storage_weekly.csv:"); print(df_to_save_totals.head())
            print("Last few rows of EIAtotals_storage_weekly.csv:"); print(df_to_save_totals.tail())
        except Exception as e:
            print(f"Error saving Totals CSV: {e}")

        print("\nCalculating weekly changes for each series...")
        df_for_changes_calc = final_totals_df.copy() 
        if not isinstance(df_for_changes_calc.index, pd.DatetimeIndex): 
             df_for_changes_calc.index = pd.to_datetime(df_for_changes_calc.index)
             df_for_changes_calc = df_for_changes_calc.sort_index()

        df_changes_all_series = pd.DataFrame(index=df_for_changes_calc.index)

        for col_name in df_for_changes_calc.columns:
            series_numeric = pd.to_numeric(df_for_changes_calc[col_name], errors='coerce')
            change_col_name = col_name.replace(" (Bcf)", "") + " Change (Bcf)" 
            df_changes_all_series[change_col_name] = series_numeric.diff()
        
        df_changes_all_series = df_changes_all_series.dropna(axis=0, how='all') 

        if not df_changes_all_series.empty:
            changes_output_file_path = OUTPUT_DIR_GLOBAL / CHANGES_FILENAME_GLOBAL
            df_to_save_changes = df_changes_all_series.copy()
            if isinstance(df_to_save_changes.index, pd.DatetimeIndex):
                 df_to_save_changes.index = df_to_save_changes.index.strftime('%Y-%m-%d')
            df_to_save_changes.index.name = "Period"
            try:
                df_to_save_changes.to_csv(changes_output_file_path, float_format='%.0f')
                print(f"\nSuccessfully saved/updated data to: {changes_output_file_path}")
                print("First few rows of EIAchanges_storage_weekly.csv:"); print(df_to_save_changes.head())
                print("Last few rows of EIAchanges_storage_weekly.csv:"); print(df_to_save_changes.tail())
            except Exception as e:
                print(f"Error saving Changes CSV: {e}")
        else:
            print("No changes data to save.")
    elif not any_new_data_fetched and df_existing_totals is None: 
         print("No data available to process for any storage CSV files.")


if __name__ == "__main__":
    main()