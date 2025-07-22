# GridStatLoadHist.py
#
# Description:
# This script fetches historical daily load for all configured ISOs from the
# GridStatus.io API. It calculates the total energy (MWh) for each day.
# The script intelligently updates the output file by replacing the last 3 days
# of data, ensuring the history is always current without re-fetching everything.
#
# Instructions:
# 1. Ensure a .env file with your GRIDSTATUS_API_KEY is in the same directory.
# 2. Run the script: python GridStatLoadHist.py

import pandas as pd
import os
from dotenv import load_dotenv
from gridstatusio import GridStatusClient
from datetime import datetime, timedelta
from pathlib import Path
import traceback

# --- Configuration ---

# The target directory for the output CSV file.
OUTPUT_DIRECTORY = Path("C:/Users/patri/OneDrive/Desktop/Coding/TraderHelper/INFO")
OUTPUT_FILENAME = "GridStatLoadHist.csv"

# The date to start fetching from if no historical file exists.
# This can be adjusted to pull more or less history on the first run.
DEFAULT_HISTORY_START_DATE = "2025-07-01"

# Configuration for historical load datasets for each ISO.
# Corrected 'time_column' for all ISOs based on error log output.
ISO_HISTORICAL_CONFIG = {
    "PJM": {
        "dataset_id": "pjm_standardized_5_min",
        "time_column": "interval_start_utc", # Corrected
        "value_column": "load.load",
        "interval_minutes": 5,
        "timezone": "America/New_York"
    },
    "SPP": {
        "dataset_id": "spp_load",
        "time_column": "interval_start_utc", # Corrected
        "value_column": "load",
        "interval_minutes": 60,
        "timezone": "America/Chicago"
    },
    "NYISO": {
        "dataset_id": "nyiso_load",
        "time_column": "interval_start_utc", # Corrected
        "value_column": "load",
        "interval_minutes": 60,
        "timezone": "America/New_York"
    },
    "MISO": {
        "dataset_id": "miso_load",
        "time_column": "interval_start_utc", # Corrected
        "value_column": "load",
        "interval_minutes": 60,
        "timezone": "America/Chicago"
    },
    "ISONE": {
        "dataset_id": "isone_load",
        "time_column": "interval_start_utc", # Corrected
        "value_column": "load",
        "interval_minutes": 60,
        "timezone": "America/New_York"
    },
    "IESO": {
        "dataset_id": "ieso_load",
        "time_column": "interval_start_utc", # Corrected
        "value_column": "market_total_load",
        "interval_minutes": 60,
        "timezone": "America/Toronto"
    },
    "ERCOT": {
        "dataset_id": "ercot_standardized_hourly",
        "time_column": "interval_start_utc", # Corrected
        "value_column": "load.load",
        "interval_minutes": 60,
        "timezone": "America/Chicago"
    },
    "CAISO": {
        "dataset_id": "caiso_load",
        "time_column": "interval_start_utc", # Corrected
        "value_column": "load",
        "interval_minutes": 60,
        "timezone": "America/Los_Angeles"
    }
}


def initialize_client() -> GridStatusClient | None:
    """Loads the API key and initializes the GridStatusClient."""
    print("--- Initializing ---")
    load_dotenv()
    api_key = os.environ.get("GRIDSTATUS_API_KEY")
    if not api_key:
        print("❌ ERROR: GRIDSTATUS_API_KEY not found in environment variables.")
        return None
    print("✅ API Key retrieved from environment.")
    try:
        client = GridStatusClient(api_key=api_key)
        print("✅ GridStatusClient initialized successfully.")
        return client
    except Exception as e:
        print(f"❌ ERROR: Could not initialize GridStatusClient: {e}")
        return None


def fetch_historical_load(client: GridStatusClient, config: dict, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches historical load data for a given date range."""
    dataset = config["dataset_id"]
    print(f"\n[Fetching] Historical data from '{dataset}'...")
    try:
        df = client.get_dataset(dataset=dataset, start=start_date, end=end_date)
        if df.empty:
            print(f"   INFO: No data returned for {start_date} to {end_date}.")
        else:
            print(f"   ✅ Successfully fetched {len(df)} rows.")
        return df
    except Exception as e:
        print(f"   ❌ ERROR while fetching data: {e}")
        return pd.DataFrame()


def process_historical_data(df: pd.DataFrame, iso: str, config: dict) -> pd.DataFrame:
    """Validates, calculates, and formats the total daily historical load."""
    if df.empty:
        return pd.DataFrame()

    print(f"[Processing] Data for {iso}...")
    time_col = config["time_column"]
    value_col_name = config["value_column"]
    interval_min = config["interval_minutes"]
    market_tz = config["timezone"]

    # Check for required columns before proceeding to prevent KeyErrors.
    if time_col not in df.columns or value_col_name not in df.columns:
        if time_col not in df.columns:
            print(f"   ❌ ERROR: Expected time column '{time_col}' not found for {iso}.")
        if value_col_name not in df.columns:
            print(f"   ❌ ERROR: Expected value column '{value_col_name}' not found for {iso}.")
        
        print(f"   ℹ️  Available columns are: {df.columns.tolist()}")
        return pd.DataFrame()

    # Data Cleaning and Preparation
    df[time_col] = pd.to_datetime(df[time_col])
    df[value_col_name] = pd.to_numeric(df[value_col_name], errors='coerce')
    df.dropna(subset=[time_col, value_col_name], inplace=True)

    # Timezone conversion to group by correct market day
    df['market_time'] = df[time_col].dt.tz_convert(market_tz)
    df['market_date'] = df['market_time'].dt.date

    # --- Data Validation: Ensure only full days are processed ---
    expected_intervals_per_day = (24 * 60) // interval_min
    daily_counts = df.groupby('market_date')[time_col].count()
    full_days_index = daily_counts[daily_counts >= expected_intervals_per_day].index
    validated_df = df[df['market_date'].isin(full_days_index)]

    if len(daily_counts) > len(full_days_index):
        partial_days_count = len(daily_counts) - len(full_days_index)
        print(f"   ⚠️  Discarded {partial_days_count} partial day(s) for {iso} with incomplete data.")

    if validated_df.empty:
        print(f"   INFO: No full days of data found for {iso} after validation.")
        return pd.DataFrame()

    # --- Calculation on Validated Data ---
    mwh_col = 'historical_mwh'
    validated_df.loc[:, mwh_col] = validated_df[value_col_name] * (interval_min / 60)
    daily_total_df = validated_df.groupby('market_date')[mwh_col].sum().reset_index()
    
    # --- Formalize Output ---
    daily_total_df["Item"] = f"{iso} - DailyTotLoad"
    daily_total_df.rename(columns={'market_date': "Date", mwh_col: "Value"}, inplace=True)
    final_df = daily_total_df[["Date", "Item", "Value"]]
    
    print(f"   ✅ Data for {iso} processed and formatted successfully.")
    return final_df


def main():
    """Main execution function to run the historical load retrieval for all ISOs."""
    client = initialize_client()
    if not client:
        print("\n--- Script cannot continue. ---")
        return

    # --- Determine Date Range for Fetching ---
    full_path = OUTPUT_DIRECTORY / OUTPUT_FILENAME
    existing_df = pd.DataFrame()
    
    if full_path.exists():
        print(f"\n--- Found existing file: {full_path} ---")
        existing_df = pd.read_csv(full_path, parse_dates=["Date"])
        # Fetch the last 3 days to get any revisions
        fetch_start_date = (datetime.now() - timedelta(days=3)).date()
        print(f"Updating data from {fetch_start_date} onwards.")
    else:
        print("\n--- No existing file found. Building full history. ---")
        fetch_start_date = datetime.strptime(DEFAULT_HISTORY_START_DATE, "%Y-%m-%d").date()
        print(f"Fetching all data from {fetch_start_date} onwards.")
        
    fetch_end_date = datetime.now().date()

    # --- Loop, Fetch, and Process ---
    all_new_results = []
    for iso, config in ISO_HISTORICAL_CONFIG.items():
        raw_df = fetch_historical_load(client, config, str(fetch_start_date), str(fetch_end_date))
        results_df = process_historical_data(raw_df, iso, config)
        if not results_df.empty:
            all_new_results.append(results_df)

    # --- Combine and Save Results ---
    if not all_new_results:
        print("\n--- No new data fetched. File is up-to-date. ---")
        return
        
    new_data_df = pd.concat(all_new_results, ignore_index=True)
    new_data_df["Date"] = pd.to_datetime(new_data_df["Date"]).dt.date
    
    # Remove data from the existing file that will be replaced
    if not existing_df.empty:
        existing_df["Date"] = pd.to_datetime(existing_df["Date"]).dt.date
        existing_df = existing_df[existing_df["Date"] < fetch_start_date]

    # Combine the old history with the newly fetched data
    combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
    combined_df.sort_values(by=["Item", "Date"], inplace=True)
    
    # Save the final, combined data
    print("\n[Saving] Results to file...")
    try:
        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(full_path, index=False)
        print(f"   ✅ Results successfully saved to: {full_path}")
    except Exception as e:
        print(f"   ❌ ERROR saving file: {e}")


if __name__ == "__main__":
    main()
    print("\n--- Script Finished ---")
