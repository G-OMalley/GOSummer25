# GSLoadFuelHist.py

import pandas as pd
import os
from dotenv import load_dotenv
from gridstatusio import GridStatusClient
from datetime import datetime, timedelta, timezone
import traceback

# --- Global Variables for Client and API Key Status ---
API_KEY_LOADED = False
GS_CLIENT = None

# --- Load API Key and Initialize Client ---
try:
    # Assumes .env file is in the same directory as this script (TraderHelper/Power/.env)
    # If running from TraderHelper/ and .env is in TraderHelper/, load_dotenv() should find it.
    # For robustness if CWD is TraderHelper/Power/ explicitly:
    # dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    # load_dotenv(dotenv_path=dotenv_path)
    load_dotenv() # Should find .env in current working dir (Power/) or parent (TraderHelper/)
    
    GRIDSTATUS_API_KEY = os.environ.get("GRIDSTATUS_API_KEY")

    if GRIDSTATUS_API_KEY:
        print("API Key retrieved from environment.")
        API_KEY_LOADED = True
        try:
            GS_CLIENT = GridStatusClient(api_key=GRIDSTATUS_API_KEY)
            print("GridStatusClient initialized successfully.")
        except Exception as e_client:
            print(f"Error initializing GridStatusClient: {e_client}")
            GS_CLIENT = None # Ensure client is None if init fails
            # traceback.print_exc()
    else:
        print("ERROR: GRIDSTATUS_API_KEY not found in environment variables.")
        print("Ensure your .env file is correctly placed (e.g., in TraderHelper/Power/ or TraderHelper/)")
        print("and contains: GRIDSTATUS_API_KEY='your_key_here'")

except Exception as e_dotenv:
    print(f"Error during .env loading or initial setup: {e_dotenv}")
    # traceback.print_exc()

# --- ISO Configurations for Analysis ---
ANALYSIS_ISO_CONFIGS = {
    "CAISO": {
        "load": {
            "dataset_id": "caiso_load_hourly",
            "value_column_per_location": "load",
            "location_identifier_column": "tac_area_name",
            "sum_locations_for_iso_total": True,
            "time_column": "interval_start_utc"
        },
        "fuel_mix": {
            "dataset_id": "caiso_fuel_mix",
            "value_columns": [
                'solar', 'wind', 'geothermal', 'biomass', 'biogas', 'small_hydro',
                'coal', 'nuclear', 'natural_gas', 'large_hydro', 'batteries',
                'imports', 'other'
            ],
            "time_column": "interval_start_utc"
        }
    },
    "ISONE": { # Basic structure for ISONE, verify column names from your CSV or API exploration
         "load": {
            "dataset_id": "isone_load_hourly", # From your CSV
            "value_column_per_location": "load", # Verify: CSV shows 'load', 'native_load', 'ard_demand'
            "location_identifier_column": "location", # From your CSV
            "sum_locations_for_iso_total": True, # Assumption: need to sum these locations
            "time_column": "interval_start_utc" # From your CSV
        },
        "fuel_mix": {
            "dataset_id": "isone_fuel_mix", # From your CSV
            "value_columns": [ # From your CSV
                'coal', 'hydro', 'landfill_gas', 'natural_gas', 'nuclear', 'oil',
                'other', 'refuse', 'solar', 'wind', 'wood'
                ],
            "time_column": "interval_start_utc" # From your CSV
        }
    }
    # Add other ISOs (ERCOT, MISO, NYISO, PJM, SPP) here as needed,
    # following the patterns and using your CSV data.
}

# --- Helper Function: Fetch General Data from GridStatus API ---
def fetch_general_data(
    client: "GridStatusClient",
    iso_name_logging: str,
    dataset_id: str,
    start_date_str: str,
    end_date_str: str,
    filter_column: str = None,
    filter_value = None, # Can be single value or list
    filter_operator: str = None,
    limit: int = None,
    verbose: bool = True
) -> pd.DataFrame:
    fetch_for_what = f"{iso_name_logging} - {dataset_id}"
    actual_filter_value_for_log = filter_value
    operator_to_use = filter_operator

    if filter_column and filter_value is not None:
        if not operator_to_use:
            operator_to_use = "in" if isinstance(filter_value, list) and len(filter_value) != 1 else "="
        
        # API sometimes prefers '=' for single item list even if 'in' is logical
        if isinstance(filter_value, list) and len(filter_value) == 1:
            if operator_to_use == "in": # If 'in' specified or defaulted for a single item list
                operator_to_use = "="   # Change to '='
            actual_filter_value_for_log = filter_value[0] # Log the single item
        
        fetch_for_what += f", Filter: {filter_column} {operator_to_use} {actual_filter_value_for_log}"

    if verbose:
        print(f"  Fetching data for {fetch_for_what}, Period: {start_date_str} to {end_date_str}")

    df_result = pd.DataFrame()
    try:
        params = {
            "dataset": dataset_id,
            "start": start_date_str,
            "end": end_date_str,
            "limit": limit
        }
        if filter_column and filter_value is not None:
            params["filter_column"] = filter_column
            
            current_filter_value = filter_value
            # Determine operator if not explicitly set
            if not filter_operator:
                params["filter_operator"] = "in" if isinstance(current_filter_value, list) and len(current_filter_value) > 1 else "="
            else:
                params["filter_operator"] = filter_operator
            
            # Adjust filter_value for API: if it's a list of one and operator became "=", pass the single item
            if isinstance(current_filter_value, list) and len(current_filter_value) == 1 and params["filter_operator"] == "=":
                 params["filter_value"] = current_filter_value[0]
            else:
                params["filter_value"] = current_filter_value


        data_response = client.get_dataset(**params)

        if isinstance(data_response, pd.DataFrame):
            df_result = data_response
        elif verbose:
            print(f"  WARNING: Unexpected data format for {fetch_for_what}: {type(data_response)}")

        if df_result.empty and verbose:
            print(f"  INFO: No data retrieved for {fetch_for_what} (this might be normal for specific filters/times).")
        elif verbose and not df_result.empty:
            print(f"  Successfully fetched {len(df_result)} rows for {fetch_for_what}.")

    except Exception as e:
        if verbose:
            print(f"  ERROR fetching data for {fetch_for_what}: {e}")
            # traceback.print_exc()
    return df_result

# --- Main Analysis Function ---
def analyze_peak_load_and_fuel_mix_for_iso(
    client: "GridStatusClient",
    iso_name: str,
    start_date_str: str,
    end_date_str: str,
    verbose: bool = True
) -> pd.DataFrame:
    print(f"\n--- Starting Peak Load and Fuel Mix Analysis for {iso_name} ---")
    print(f"Analysis Period: {start_date_str} to {end_date_str} (inclusive)")

    if iso_name not in ANALYSIS_ISO_CONFIGS:
        print(f"ERROR: Configuration for {iso_name} not found in ANALYSIS_ISO_CONFIGS.")
        return pd.DataFrame()

    iso_config = ANALYSIS_ISO_CONFIGS[iso_name]
    load_config = iso_config.get("load")
    fm_config = iso_config.get("fuel_mix")

    if not load_config or not fm_config:
        print(f"ERROR: Incomplete load or fuel_mix configuration for {iso_name}.")
        return pd.DataFrame()

    api_fetch_end_date = (datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    if verbose: print(f"\n[Step 1] Fetching hourly load data for {iso_name}...")
    df_load_raw = fetch_general_data(
        client=client,
        iso_name_logging=iso_name,
        dataset_id=load_config["dataset_id"],
        start_date_str=start_date_str,
        end_date_str=api_fetch_end_date,
        verbose=verbose
    )

    if df_load_raw.empty:
        print(f"No load data found for {iso_name} for the period. Cannot proceed.")
        return pd.DataFrame()

    load_time_col = load_config["time_column"]
    if load_time_col not in df_load_raw.columns:
        print(f"CRITICAL: Load time column '{load_time_col}' not found in {iso_name} data. Columns: {df_load_raw.columns.tolist()}")
        return pd.DataFrame()
    try:
        df_load_raw[load_time_col] = pd.to_datetime(df_load_raw[load_time_col], errors='coerce')
        df_load_raw.dropna(subset=[load_time_col], inplace=True)
    except Exception as e_time_conv:
        print(f"CRITICAL: Error converting load time column '{load_time_col}' to datetime: {e_time_conv}")
        return pd.DataFrame()


    if verbose: print(f"\n[Step 2] Processing load data to calculate total hourly ISO load for {iso_name}...")
    df_iso_hourly_load = pd.DataFrame()

    if load_config.get("sum_locations_for_iso_total", False):
        loc_col = load_config["location_identifier_column"]
        val_col = load_config["value_column_per_location"]
        if loc_col not in df_load_raw.columns or val_col not in df_load_raw.columns:
            print(f"CRITICAL: Load columns ('{loc_col}', '{val_col}') for summing not found. Columns: {df_load_raw.columns.tolist()}")
            return pd.DataFrame()
        
        df_load_raw[val_col] = pd.to_numeric(df_load_raw[val_col], errors='coerce')
        df_load_raw.dropna(subset=[val_col], inplace=True)

        unique_locations = df_load_raw[loc_col].unique()
        if verbose:
            print(f"  Found {len(unique_locations)} unique items in load location column '{loc_col}'.")
            if iso_name in ["CAISO", "ISONE"]:
                print(f"  Unique '{loc_col}' values for {iso_name} (first 20 shown): {sorted(list(map(str,unique_locations)))[:20]}")
        
        df_iso_hourly_load = df_load_raw.groupby(load_time_col)[val_col].sum().reset_index()
        df_iso_hourly_load.rename(columns={val_col: "total_iso_load", load_time_col: "timestamp_utc"}, inplace=True)
    elif "value_column_iso_total" in load_config:
        val_col_total = load_config["value_column_iso_total"]
        if val_col_total not in df_load_raw.columns:
            print(f"CRITICAL: Total ISO load column '{val_col_total}' not found. Columns: {df_load_raw.columns.tolist()}")
            return pd.DataFrame()
        df_load_raw[val_col_total] = pd.to_numeric(df_load_raw[val_col_total], errors='coerce')
        df_load_raw.dropna(subset=[val_col_total], inplace=True)
        df_iso_hourly_load = df_load_raw[[load_time_col, val_col_total]].copy()
        df_iso_hourly_load.rename(columns={val_col_total: "total_iso_load", load_time_col: "timestamp_utc"}, inplace=True)
    else:
        print(f"CRITICAL: Load configuration for {iso_name} is unclear (needs 'sum_locations_for_iso_total' or 'value_column_iso_total').")
        return pd.DataFrame()

    if df_iso_hourly_load.empty:
        print(f"Could not derive total hourly ISO load for {iso_name}.")
        return pd.DataFrame()

    if verbose: print(f"  Total ISO hourly load calculated. Shape: {df_iso_hourly_load.shape}")


    if verbose: print(f"\n[Step 3] Finding daily maximum load and corresponding peak hour for {iso_name}...")
    df_iso_hourly_load['total_iso_load'] = pd.to_numeric(df_iso_hourly_load['total_iso_load'], errors='coerce')
    df_iso_hourly_load.dropna(subset=['total_iso_load'], inplace=True)
    if df_iso_hourly_load.empty:
        print(f"Total hourly ISO load data is empty after numeric conversion/NaN drop for {iso_name}.")
        return pd.DataFrame()

    df_iso_hourly_load['date_utc'] = df_iso_hourly_load['timestamp_utc'].dt.date
    daily_peak_indices = df_iso_hourly_load.groupby('date_utc')['total_iso_load'].idxmax()
    daily_peaks_df = df_iso_hourly_load.loc[daily_peak_indices].copy() # Use .copy() to avoid SettingWithCopyWarning
    daily_peaks_df.rename(columns={
        'timestamp_utc': 'peak_load_hour_utc',
        'total_iso_load': 'peak_iso_load_mw'
    }, inplace=True)

    if verbose:
        print(f"  Identified {len(daily_peaks_df)} daily peak load hours.")
        if not daily_peaks_df.empty: print(daily_peaks_df[['date_utc', 'peak_load_hour_utc', 'peak_iso_load_mw']].head().to_string())
    if daily_peaks_df.empty:
        print(f"No daily peaks identified for {iso_name}.")
        return pd.DataFrame()

    if verbose: print(f"\n[Step 4] Fetching fuel mix data for {iso_name} at each peak load hour...")
    fm_time_col = fm_config["time_column"]
    fm_dataset_id = fm_config["dataset_id"]
    fm_value_cols_config = fm_config["value_columns"]
    all_peak_combined_data = []

    for _, peak_row in daily_peaks_df.iterrows():
        peak_hour_start_utc = peak_row['peak_load_hour_utc']
        peak_date_obj = peak_row['date_utc'] # This is a datetime.date object

        if verbose: print(f"  Querying fuel mix for {peak_date_obj.strftime('%Y-%m-%d')} at peak hour: {peak_hour_start_utc.strftime('%H:%M:%S')} UTC")
        fm_fetch_start_str = peak_hour_start_utc.strftime("%Y-%m-%dT%H:%M:%S")
        fm_fetch_end_str = (peak_hour_start_utc + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")

        df_fm_for_hour = fetch_general_data(
            client=client, iso_name_logging=iso_name, dataset_id=fm_dataset_id,
            start_date_str=fm_fetch_start_str, end_date_str=fm_fetch_end_str,
            verbose=False
        )
        current_peak_data = {
            'date': peak_date_obj, # Store as date object
            'peak_load_hour_utc': peak_hour_start_utc,
            'peak_iso_load_mw': peak_row['peak_iso_load_mw']
        }
        for fm_val_col in fm_value_cols_config: current_peak_data[f"{fm_val_col}_at_peak_mw"] = pd.NA

        if not df_fm_for_hour.empty:
            if fm_time_col not in df_fm_for_hour.columns:
                if verbose: print(f"    WARNING: Fuel mix time col '{fm_time_col}' not found at {peak_hour_start_utc}.")
            else:
                try:
                    df_fm_for_hour[fm_time_col] = pd.to_datetime(df_fm_for_hour[fm_time_col], errors='coerce')
                    exact_hour_fm_data = df_fm_for_hour[df_fm_for_hour[fm_time_col] == peak_hour_start_utc].copy()
                    if not exact_hour_fm_data.empty:
                        fm_record = exact_hour_fm_data.iloc[0]
                        for fm_val_col in fm_value_cols_config:
                            if fm_val_col in fm_record.index:
                                current_peak_data[f"{fm_val_col}_at_peak_mw"] = fm_record[fm_val_col]
                        if verbose: print(f"    Processed fuel mix for {peak_hour_start_utc}.")
                    elif verbose: print(f"    No fuel mix data for precise peak hour {peak_hour_start_utc} after filtering.")
                except Exception as e_fm_proc:
                    if verbose: print(f"    Error processing fuel mix for {peak_hour_start_utc}: {e_fm_proc}")
        elif verbose: print(f"    No fuel mix data fetched from API for {peak_hour_start_utc}.")
        all_peak_combined_data.append(current_peak_data)

    if not all_peak_combined_data:
        print(f"No fuel mix data aligned with peak load hours for {iso_name}.")
        return daily_peaks_df if not daily_peaks_df.empty else pd.DataFrame()

    results_df = pd.DataFrame(all_peak_combined_data)
    final_ordered_columns = ['date', 'peak_load_hour_utc', 'peak_iso_load_mw']
    for fm_val_col_config in fm_config["value_columns"]:
        result_col_name = f"{fm_val_col_config}_at_peak_mw"
        if result_col_name not in results_df.columns: results_df[result_col_name] = pd.NA
        final_ordered_columns.append(result_col_name)
    final_ordered_columns.extend([col for col in results_df.columns if col not in final_ordered_columns])
    results_df = results_df[final_ordered_columns]

    if verbose:
        print(f"\n[Step 5] Combined Peak Load and Fuel Mix Results for {iso_name}:")
        print(results_df.to_string(max_rows=10))
    return results_df

# --- Main Execution Block ---
if __name__ == "__main__":
    if API_KEY_LOADED and GS_CLIENT:
        print("\n--- Starting Peak Load & Fuel Mix Analysis Script ---")
        
        # --- User Inputs for ISO and Date Range ---
        available_isos = list(ANALYSIS_ISO_CONFIGS.keys())
        print(f"Available ISOs for analysis: {', '.join(available_isos)}")
        
        target_iso = ""
        while target_iso not in available_isos:
            target_iso_input = input(f"Enter ISO to analyze (e.g., {available_isos[0]}): ").strip().upper()
            if target_iso_input in available_isos:
                target_iso = target_iso_input
            else:
                print(f"Invalid ISO. Please choose from: {', '.join(available_isos)}")

        # Date range input with defaults
        default_end_date = datetime.now(timezone.utc).date() - timedelta(days=2) # day before yesterday (more likely to have full data)
        default_start_date = default_end_date - timedelta(days=2) # 3 days total

        start_date_input_str = input(f"Enter Start Date (YYYY-MM-DD) [default: {default_start_date.strftime('%Y-%m-%d')}]: ").strip() or default_start_date.strftime('%Y-%m-%d')
        end_date_input_str = input(f"Enter End Date (YYYY-MM-DD, inclusive) [default: {default_end_date.strftime('%Y-%m-%d')}]: ").strip() or default_end_date.strftime('%Y-%m-%d')
        
        try:
            # Validate dates
            start_dt = datetime.strptime(start_date_input_str, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date_input_str, "%Y-%m-%d")
            if end_dt < start_dt:
                print("ERROR: End date cannot be before start date.")
            else:
                # Call the main analysis function
                results_df = analyze_peak_load_and_fuel_mix_for_iso(
                    client=GS_CLIENT,
                    iso_name=target_iso,
                    start_date_str=start_date_input_str,
                    end_date_str=end_date_input_str,
                    verbose=True
                )

                if not results_df.empty:
                    # Save to CSV
                    safe_iso = "".join(c if c.isalnum() else '_' for c in target_iso)
                    safe_start = "".join(c if c.isalnum() else '_' for c in start_date_input_str)
                    safe_end = "".join(c if c.isalnum() else '_' for c in end_date_input_str)
                    
                    # Ensure CSV is saved in the same directory as the script or a sub-directory
                    output_dir = os.path.dirname(__file__) # Gets directory of current script
                    csv_filename = f"{safe_iso}_peak_load_fuel_mix_{safe_start}_to_{safe_end}.csv"
                    full_csv_path = os.path.join(output_dir, csv_filename)
                    
                    try:
                        results_df.to_csv(full_csv_path, index=False)
                        print(f"\nSUCCESS: Results saved to: {full_csv_path}")
                    except Exception as e_csv:
                        print(f"ERROR saving results to CSV '{full_csv_path}': {e_csv}")
                else:
                    print(f"\nNo results generated for {target_iso} for the period {start_date_input_str} to {end_date_input_str}.")

        except ValueError:
            print("ERROR: Invalid date format entered. Please use YYYY-MM-DD.")
        except Exception as e_main:
            print(f"An unexpected error occurred in the main execution block: {e_main}")
            traceback.print_exc()
    else:
        print("Script cannot run: GridStatus API Key or Client was not initialized.")

    print("\n--- Analysis Script Finished ---")