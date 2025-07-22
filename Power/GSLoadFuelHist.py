# GridStatLoadForecast.py
#
# Description:
# This script fetches the latest daily load forecast for all configured ISOs
# from the GridStatus.io API. It processes the data to calculate the total
# forecasted energy (MWh) for each day and saves the combined result to a CSV file.
# It validates that only full days of data are included in the totals.
#
# Instructions:
# 1. Ensure a .env file with your GRIDSTATUS_API_KEY is in the same directory.
# 2. Run the script: python GridStatLoadForecast.py

import pandas as pd
import os
from dotenv import load_dotenv
from gridstatusio import GridStatusClient
from datetime import datetime, timedelta
from pathlib import Path
import traceback

# --- Configuration ---

# The target directory for the output CSV files.
OUTPUT_DIRECTORY = Path("C:/Users/patri/OneDrive/Desktop/Coding/TraderHelper/INFO")

# Configuration for forecast datasets for each ISO.
ISO_FORECAST_CONFIG = {
    "PJM": {
        "dataset_id": "pjm_load_forecast",
        "time_column": "interval_start_utc",
        "value_column": "load_forecast",
        "interval_minutes": 5,               # PJM data is in 5-minute intervals
        "timezone": "America/New_York"       # Market timezone for PJM
    },
    "SPP": {
        "dataset_id": "spp_load_forecast",
        "time_column": "interval_start_utc",
        "value_column": "load_forecast",
        "interval_minutes": 60,              # SPP data is typically hourly
        "timezone": "America/Chicago"        # Market timezone for SPP (Central)
    },
    "NYISO": {
        "dataset_id": "nyiso_load_forecast",
        "time_column": "interval_start_utc",
        "value_column": "load_forecast",
        "interval_minutes": 5,               # NYISO data is in 5-minute intervals
        "timezone": "America/New_York"       # Market timezone for NYISO
    },
    "MISO": {
        "dataset_id": "miso_load_forecast",
        "time_column": "interval_start_utc",
        "value_column": "load_forecast",
        "interval_minutes": 5,               # MISO data is in 5-minute intervals
        "timezone": "America/Chicago"        # MISO is primarily Central Time
    },
    "ISONE": {
        "dataset_id": "isone_load_forecast_hourly",
        "time_column": "interval_start_utc",
        "value_column": "load_forecast",
        "interval_minutes": 60,              # ISONE data is hourly
        "timezone": "America/New_York"       # Market timezone for ISONE
    },
    "IESO": {
        "dataset_id": "ieso_load_forecast_hourly",
        "time_column": "interval_start_utc",
        "value_column": "ontario_load_forecast", # Specific column name for IESO
        "interval_minutes": 60,                 # IESO data is hourly
        "timezone": "America/Toronto"           # Market timezone for IESO (Ontario)
    },
    "ERCOT": {
        "dataset_id": "ercot_load_forecast",
        "time_column": "interval_start_utc",
        "value_column": "load_forecast",
        "interval_minutes": 5,               # ERCOT data is in 5-minute intervals
        "timezone": "America/Chicago"        # Market timezone for ERCOT (Central)
    },
    "CAISO": {
        "dataset_id": "caiso_load_forecast",
        "time_column": "interval_start_utc",
        "value_column": "load_forecast",
        "interval_minutes": 5,               # CAISO data is in 5-minute intervals
        "timezone": "America/Los_Angeles"    # Market timezone for CAISO (Pacific)
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


def fetch_daily_load_forecast(client: GridStatusClient, config: dict, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches the latest load forecast data for a given date range."""
    dataset = config["dataset_id"]
    print(f"\n[Fetching] Latest forecast from '{dataset}'...")
    
    try:
        df = client.get_dataset(
            dataset=dataset,
            start=start_date,
            end=end_date,
            publish_time="latest"
        )
        if df.empty:
            print(f"   INFO: No forecast data returned for {start_date} to {end_date}.")
        else:
            print(f"   ✅ Successfully fetched {len(df)} forecast rows.")
        return df
    except Exception as e:
        print(f"   ❌ ERROR while fetching forecast data: {e}")
        traceback.print_exc()
        return pd.DataFrame()


def process_forecast_data(df: pd.DataFrame, iso: str, config: dict) -> pd.DataFrame:
    """
    Validates for full-day data, then calculates and formats the total forecasted energy (MWh).
    """
    if df.empty:
        return pd.DataFrame()

    print(f"[Processing] Data for {iso}...")
    
    time_col = config["time_column"]
    value_col = config["value_column"]
    interval_min = config["interval_minutes"]
    market_tz = config["timezone"]

    if time_col not in df.columns or value_col not in df.columns:
        print(f"   ❌ ERROR: Expected columns '{time_col}' and '{value_col}' not found in data.")
        print(f"   Available columns are: {df.columns.tolist()}")
        return pd.DataFrame()

    # Data Cleaning and Preparation
    df[time_col] = pd.to_datetime(df[time_col])
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df.dropna(subset=[time_col, value_col], inplace=True)

    # Timezone conversion to group by correct market day
    df['market_time'] = df[time_col].dt.tz_convert(market_tz)
    df['market_date'] = df['market_time'].dt.date

    # --- Data Validation: Ensure only full days are processed ---
    expected_intervals_per_day = (24 * 60) // interval_min
    
    # Count the number of intervals recorded for each day.
    daily_counts = df.groupby('market_date')[time_col].count().rename('interval_count').reset_index()
    
    # Identify which days have a full set of data.
    # Using >= handles edge cases like DST, but == is stricter. We'll use >= for robustness.
    full_days = daily_counts[daily_counts['interval_count'] >= expected_intervals_per_day]
    
    # Filter the original dataframe to keep only the full days.
    validated_df = df[df['market_date'].isin(full_days['market_date'])]

    if len(daily_counts) > len(full_days):
        partial_days_count = len(daily_counts) - len(full_days)
        print(f"   ⚠️  Discarded {partial_days_count} partial day(s) for {iso} with incomplete data.")
        print(f"      (Expected at least {expected_intervals_per_day} intervals per day)")

    if validated_df.empty:
        print(f"   INFO: No full days of data found for {iso} after validation.")
        return pd.DataFrame()
        
    # --- Calculation on Validated Data ---
    # Calculate MWh for each interval
    mwh_col = 'load_forecast_mwh'
    # Use .loc to avoid SettingWithCopyWarning
    validated_df.loc[:, mwh_col] = validated_df[value_col] * (interval_min / 60)
    
    # Group by market date and sum to get daily total
    daily_total_df = validated_df.groupby('market_date')[mwh_col].sum().reset_index()
    
    # --- Formalize Output ---
    daily_total_df["Item"] = f"{iso} - TotMWForecast"
    daily_total_df.rename(columns={'market_date': "Forecast Date", mwh_col: "Value"}, inplace=True)
    final_df = daily_total_df[["Forecast Date", "Item", "Value"]]
    
    print(f"   ✅ Data for {iso} processed and formatted successfully.")
    return final_df


def save_results_to_file(all_forecasts_df: pd.DataFrame):
    """
    Appends or updates the forecast data for all processed ISOs in the target CSV file.
    If the file doesn't exist, it's created.
    """
    if all_forecasts_df.empty:
        return
        
    print("\n[Saving] Results to file...")
    
    try:
        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        filename = "GridStatLoadForecast.csv"
        full_path = OUTPUT_DIRECTORY / filename

        existing_df = pd.DataFrame()
        if full_path.exists():
            print(f"   - Found existing file at {full_path}. Reading...")
            existing_df = pd.read_csv(full_path)
            # Get list of all items (ISOs) in the new data
            items_to_update = all_forecasts_df['Item'].unique()
            # Remove any old data for the current ISOs to avoid duplicates
            existing_df = existing_df[~existing_df['Item'].isin(items_to_update)]
            print(f"   - Updating data for: {', '.join(items_to_update)}.")
        
        # Combine the old data (for other ISOs) with the new data
        combined_df = pd.concat([existing_df, all_forecasts_df], ignore_index=True)
        
        # Sort by Item and then by date for a clean, predictable file layout
        combined_df.sort_values(by=["Item", "Forecast Date"], inplace=True)
        
        combined_df.to_csv(full_path, index=False)
        print(f"   ✅ Results successfully saved to: {full_path}")

    except Exception as e:
        print(f"   ❌ ERROR saving file: {e}")


def main():
    """Main execution function to run the forecast retrieval for all ISOs."""
    client = initialize_client()
    if not client:
        print("\n--- Script cannot continue. Please fix initialization errors. ---")
        return

    # --- Date Range ---
    today = datetime.now()
    default_start = today
    default_end = today + timedelta(days=7)
    start_date_str = default_start.strftime('%Y-%m-%d')
    end_date_str = default_end.strftime('%Y-%m-%d')
    print(f"\n--- Running Forecast Analysis for all ISOs ---")
    print(f"Fetching forecast for period: {start_date_str} to {end_date_str}")
    
    all_results = []
    # --- Loop through all configured ISOs ---
    for iso, config in ISO_FORECAST_CONFIG.items():
        raw_df = fetch_daily_load_forecast(client, config, start_date_str, end_date_str)
        results_df = process_forecast_data(raw_df, iso, config)
        if not results_df.empty:
            all_results.append(results_df)

    # --- Combine and Save ---
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        print("\n--- Combined Forecast Results ---")
        print(final_df.to_string(index=False))
        save_results_to_file(final_df)
    else:
        print("\n--- Analysis finished with no results to display. ---")


if __name__ == "__main__":
    main()
    print("\n--- Script Finished ---")