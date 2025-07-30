# GridStatLoadAndForecast.py
#
# Description:
# This script fetches both historical and forecast daily load for all configured ISOs
# from the GridStatus.io API. It calculates the total energy (MWh) for each day.
#
# Key Feature: Split-Day Handling
# For the most recent day with data, if historical data is incomplete, the script
# "stitches" it together with forecast data for the remaining hours of that same
# day. This complete, stitched day is treated as historical "TotalLoad". The
# dedicated forecast file then begins on the following day.
#
# Instructions:
# 1. Ensure a .env file with your GRIDSTATUS_API_KEY is in the same directory.
# 2. Run the script: python GridStatLoadAndForecast.py

import pandas as pd
import os
from dotenv import load_dotenv
from gridstatusio import GridStatusClient
from datetime import datetime, timedelta
from pathlib import Path

# --- Configuration ---
OUTPUT_DIRECTORY = Path("C:/Users/patri/OneDrive/Desktop/Coding/TraderHelper/INFO")
DEFAULT_HISTORY_START_DATE = "2025-07-01"

# --- ISO Configurations ---

# ** CHANGE: Updated ERCOT config to use 15-minute data for better accuracy. **
ISO_HISTORICAL_CONFIG = {
    "PJM": {"dataset_id": "pjm_standardized_5_min", "time_column": "interval_start_utc", "value_column": "load.load", "interval_minutes": 5, "timezone": "America/New_York"},
    "SPP": {"dataset_id": "spp_load", "time_column": "interval_start_utc", "value_column": "load", "interval_minutes": 60, "timezone": "America/Chicago"},
    "NYISO": {"dataset_id": "nyiso_load", "time_column": "interval_start_utc", "value_column": "load", "interval_minutes": 60, "timezone": "America/New_York"},
    "MISO": {"dataset_id": "miso_load", "time_column": "interval_start_utc", "value_column": "load", "interval_minutes": 60, "timezone": "America/Chicago"},
    "ISONE": {"dataset_id": "isone_load", "time_column": "interval_start_utc", "value_column": "load", "interval_minutes": 60, "timezone": "America/New_York"},
    "IESO": {"dataset_id": "ieso_load", "time_column": "interval_start_utc", "value_column": "market_total_load", "interval_minutes": 60, "timezone": "America/Toronto"},
    "ERCOT": {"dataset_id": "ercot_load_15_min", "time_column": "interval_start_utc", "value_column": "demand", "interval_minutes": 15, "timezone": "America/Chicago"},
    "CAISO": {"dataset_id": "caiso_load", "time_column": "interval_start_utc", "value_column": "load", "interval_minutes": 60, "timezone": "America/Los_Angeles"}
}

ISO_FORECAST_CONFIG = {
    "PJM": {"dataset_id": "pjm_load_forecast", "time_column": "interval_start_utc", "value_column": "load_forecast", "interval_minutes": 60, "timezone": "America/New_York"},
    "SPP": {"dataset_id": "spp_load_forecast", "time_column": "interval_start_utc", "value_column": "load_forecast", "interval_minutes": 60, "timezone": "America/Chicago"},
    "NYISO": {"dataset_id": "nyiso_load_forecast", "time_column": "interval_start_utc", "value_column": "load_forecast", "interval_minutes": 60, "timezone": "America/New_York"},
    "MISO": {"dataset_id": "miso_load_forecast", "time_column": "interval_start_utc", "value_column": "load_forecast", "interval_minutes": 60, "timezone": "America/Chicago"},
    "ISONE": {"dataset_id": "isone_load_forecast", "time_column": "interval_start_utc", "value_column": "load_forecast", "interval_minutes": 60, "timezone": "America/New_York"},
    # ** CHANGE: IESO forecast is currently unavailable via API, commented out to prevent errors. **
    # "IESO": {"dataset_id": "ieso_load_forecast", "time_column": "interval_start_utc", "value_column": "load_forecast", "interval_minutes": 60, "timezone": "America/Toronto"},
    "ERCOT": {"dataset_id": "ercot_load_forecast", "time_column": "interval_start_utc", "value_column": "load_forecast", "interval_minutes": 60, "timezone": "America/Chicago"},
    "CAISO": {"dataset_id": "caiso_load_forecast", "time_column": "interval_start_utc", "value_column": "load_forecast", "interval_minutes": 60, "timezone": "America/Los_Angeles"}
}

def initialize_client() -> GridStatusClient | None:
    """Initializes the GridStatusClient."""
    print("--- Initializing ---")
    load_dotenv()
    api_key = os.environ.get("GRIDSTATUS_API_KEY")
    if not api_key:
        print("❌ ERROR: GRIDSTATUS_API_KEY not found.")
        return None
    try:
        client = GridStatusClient(api_key=api_key)
        print("✅ GridStatusClient initialized successfully.")
        return client
    except Exception as e:
        print(f"❌ ERROR: Could not initialize GridStatusClient: {e}")
        return None

def fetch_data(client: GridStatusClient, dataset_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches data and handles potential errors."""
    print(f"\n[Fetching] Data from '{dataset_id}' for {start_date} to {end_date}...")
    try:
        df = client.get_dataset(dataset=dataset_id, start=start_date, end=end_date)
        print(f"   ✅ Successfully fetched {len(df)} rows.")
        return df
    except Exception as e:
        print(f"   ❌ ERROR while fetching data: {e}")
        return pd.DataFrame()

def process_daily_total(df: pd.DataFrame, config: dict, target_date: pd.Timestamp.date) -> pd.DataFrame:
    """Processes a DataFrame for a specific day to calculate total MWh."""
    time_col, value_col = config["time_column"], config["value_column"]
    interval_min = config["interval_minutes"]

    day_df = df[pd.to_datetime(df[time_col]).dt.date == target_date].copy()
    if day_df.empty:
        return pd.DataFrame()

    day_df[value_col] = pd.to_numeric(day_df[value_col], errors='coerce')
    day_df.dropna(subset=[value_col], inplace=True)

    # Check if the day is complete
    expected_intervals = (24 * 60) / interval_min
    if len(day_df) < expected_intervals:
        print(f"   ⚠️  Discarding partial day {target_date} with {len(day_df)}/{expected_intervals} intervals.")
        return pd.DataFrame()
    
    # Calculate total MWh
    day_df['mwh'] = day_df[value_col] * (interval_min / 60)
    total_mwh = day_df['mwh'].sum()

    return pd.DataFrame([{'Date': target_date, 'Value': total_mwh}])

def main():
    """Main execution function."""
    client = initialize_client()
    if not client: return

    # =================================================================
    # --- 1. PROCESS HISTORICAL DATA WITH STITCHING ---
    # =================================================================
    print("\n" + "="*25 + "\n--- Processing Historical Load Data ---\n" + "="*25)
    hist_filename = "GridStatLoadHist.csv"
    hist_path = OUTPUT_DIRECTORY / hist_filename
    
    # Determine date range for fetching historical data
    if hist_path.exists():
        print(f"--- Found existing file: {hist_path} ---")
        existing_hist_df = pd.read_csv(hist_path, parse_dates=['Date'])
        hist_start_date = (datetime.now() - timedelta(days=5)).date()
        print(f"Updating data from {hist_start_date} onwards.")
    else:
        print("--- No existing file found. Building full history. ---")
        existing_hist_df = pd.DataFrame()
        hist_start_date = datetime.strptime(DEFAULT_HISTORY_START_DATE, "%Y-%m-%d").date()
    
    hist_end_date = datetime.now().date() + timedelta(days=1) # Fetch into tomorrow to get full data for today
    
    all_final_hist = []
    
    for iso, hist_config in ISO_HISTORICAL_CONFIG.items():
        print(f"\n--- Processing {iso} ---")
        tz = hist_config['timezone']
        
        # Fetch historical data
        hist_df = fetch_data(client, hist_config['dataset_id'], str(hist_start_date), str(hist_end_date))
        if hist_df.empty:
            continue

        # Prepare dataframe by converting to local market time
        hist_df[hist_config['time_column']] = pd.to_datetime(hist_df[hist_config['time_column']]).dt.tz_convert(tz)
        hist_df['market_date'] = hist_df[hist_config['time_column']].dt.date

        # Identify dates to process
        unique_dates = sorted(hist_df['market_date'].unique())
        if not unique_dates:
            continue
            
        # Process all days except the most recent one
        for day in unique_dates[:-1]:
            day_total_df = process_daily_total(hist_df, hist_config, day)
            if not day_total_df.empty:
                all_final_hist.append(day_total_df.assign(Item=f"{iso} - TotalLoad"))

        # --- Handle the most recent day (potential split-day) ---
        latest_date = unique_dates[-1]
        latest_day_hist_df = hist_df[hist_df['market_date'] == latest_date]
        
        expected_intervals = (24 * 60) / hist_config['interval_minutes']
        
        # If the last day is complete, process it normally
        if len(latest_day_hist_df) >= expected_intervals:
            print(f"   ℹ️  Latest day {latest_date} is complete. Processing as historical.")
            day_total_df = process_daily_total(hist_df, hist_config, latest_date)
            if not day_total_df.empty:
                all_final_hist.append(day_total_df.assign(Item=f"{iso} - TotalLoad"))
        else:
            # --- Stitching Logic ---
            print(f"   ℹ️  Latest day {latest_date} is partial. Attempting to stitch with forecast...")
            fcst_config = ISO_FORECAST_CONFIG.get(iso)
            if not fcst_config:
                print(f"   ⚠️  No forecast config for {iso}. Cannot stitch. Skipping partial day.")
                continue

            # Fetch forecast for the same day
            fcst_df = fetch_data(client, fcst_config['dataset_id'], str(latest_date), str(latest_date + timedelta(days=1)))
            if fcst_df.empty:
                print(f"   ⚠️  No forecast data available for {latest_date}. Skipping partial day.")
                continue

            # Prepare forecast data
            fcst_df[fcst_config['time_column']] = pd.to_datetime(fcst_df[fcst_config['time_column']]).dt.tz_convert(tz)
            
            # Combine historical and forecast data for the single day
            hist_subset = latest_day_hist_df[[hist_config['time_column'], hist_config['value_column']]].rename(columns={hist_config['time_column']: 'time', hist_config['value_column']: 'value'})
            fcst_subset = fcst_df[[fcst_config['time_column'], fcst_config['value_column']]].rename(columns={fcst_config['time_column']: 'time', fcst_config['value_column']: 'value'})
            
            stitched_df = pd.concat([hist_subset, fcst_subset])
            stitched_df.drop_duplicates(subset='time', keep='first', inplace=True) # Keep historical over forecast
            
            # Create a temporary config for the stitched data
            stitched_config = {"time_column": "time", "value_column": "value", "interval_minutes": fcst_config['interval_minutes']}
            
            day_total_df = process_daily_total(stitched_df, stitched_config, latest_date)
            if not day_total_df.empty:
                print(f"   ✅ Successfully stitched and processed {latest_date}.")
                all_final_hist.append(day_total_df.assign(Item=f"{iso} - TotalLoad"))

    # Combine and save historical data
    if all_final_hist:
        new_hist_df = pd.concat(all_final_hist, ignore_index=True)
        new_hist_df['Date'] = pd.to_datetime(new_hist_df['Date'])
        
        # Merge with existing data, removing old rows that will be updated
        if not existing_hist_df.empty:
            final_hist_df = pd.concat([existing_hist_df[existing_hist_df['Date'].dt.date < hist_start_date], new_hist_df])
        else:
            final_hist_df = new_hist_df
            
        final_hist_df.drop_duplicates(subset=['Date', 'Item'], keep='last', inplace=True)
        final_hist_df.sort_values(by=["Item", "Date"], inplace=True)
        
        final_hist_df.to_csv(hist_path, index=False, date_format='%Y-%m-%d')
        print(f"\n✅ Historical data saved to {hist_path}")
    else:
        final_hist_df = existing_hist_df
        print("\n--- No new historical data to save. ---")


    # =================================================================
    # --- 2. PROCESS FORECAST DATA ---
    # =================================================================
    print("\n" + "="*25 + "\n--- Processing Forecast Load Data ---\n" + "="*25)
    if final_hist_df.empty:
        print("❌ Cannot run forecast: historical data is empty.")
        return

    # Forecast starts the day after the last historical/stitched day
    last_hist_date = final_hist_df['Date'].max().date()
    fcst_start_date = last_hist_date + timedelta(days=1)
    fcst_end_date = fcst_start_date + timedelta(days=14)
    print(f"Fetching forecast data from {fcst_start_date} to {fcst_end_date}")

    all_fcst_results = []
    for iso, fcst_config in ISO_FORECAST_CONFIG.items():
        fcst_df = fetch_data(client, fcst_config['dataset_id'], str(fcst_start_date), str(fcst_end_date))
        if fcst_df.empty: continue
            
        fcst_df[fcst_config['time_column']] = pd.to_datetime(fcst_df[fcst_config['time_column']]).dt.tz_convert(fcst_config['timezone'])
        fcst_df['market_date'] = fcst_df[fcst_config['time_column']].dt.date
        
        for day in sorted(fcst_df['market_date'].unique()):
            day_total_df = process_daily_total(fcst_df, fcst_config, day)
            if not day_total_df.empty:
                all_fcst_results.append(day_total_df.assign(Item=f"{iso} - ForecastLoad"))

    if all_fcst_results:
        final_fcst_df = pd.concat(all_fcst_results, ignore_index=True)
        final_fcst_df.sort_values(by=["Item", "Date"], inplace=True)
        fcst_filename = "GridStatLoadForecast.csv"
        fcst_path = OUTPUT_DIRECTORY / fcst_filename
        final_fcst_df.to_csv(fcst_path, index=False, date_format='%Y-%m-%d')
        print(f"\n✅ Forecast data saved to {fcst_path}")
    else:
        print("\n--- No new forecast data to save. ---")

if __name__ == "__main__":
    main()
    print("\n--- Script Finished ---")