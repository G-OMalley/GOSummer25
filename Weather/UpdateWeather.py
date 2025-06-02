import pandas as pd
import os
from datetime import datetime, timedelta
import requests
from io import StringIO
from dotenv import load_dotenv
import numpy as np

# --- Configuration & Constants ---
# Load credentials from .env file
load_dotenv()
USERNAME = os.getenv("WSI_ACCOUNT_USERNAME")
PROFILE = os.getenv("WSI_PROFILE_EMAIL")
PASSWORD = os.getenv("WSI_PASSWORD")

# API Base URL
BASE_SERVICE_URL = "https://www.wsitrader.com/Services/CSVDownloadService.svc"

# --- Define File Paths ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
base_project_dir = os.path.dirname(current_script_dir)
weather_csv_path = os.path.join(base_project_dir, "INFO", "WEATHER.csv")

# Define the exact 13 columns we want to keep
EXPECTED_COLUMNS = [
    'City Symbol', 'City Title', 'Date', 'Min Temp', 'Max Temp', 'Avg Temp',
    'CDD', 'HDD', '10yr Min Temp', '10yr Max Temp', '10yr Avg Temp',
    '10yr CDD', '10yr HDD'
]
# Columns to average for 10yr normals
TEN_YEAR_NORMAL_SOURCE_COLS = ['Min Temp', 'Max Temp', 'Avg Temp', 'CDD', 'HDD']
TEN_YEAR_NORMAL_TARGET_COLS = ['10yr Min Temp', '10yr Max Temp', '10yr Avg Temp', '10yr CDD', '10yr HDD']


# --- Helper Functions ---
def parse_api_response_csv(raw_csv_text, city_symbol):
    """ Parses the CSV text from WSI API, handling potential header issues. """
    if not raw_csv_text.strip() or "no data" in raw_csv_text.lower() or "Error" in raw_csv_text:
        print(f"  ⚠️ No data in raw CSV or error for {city_symbol}. Response: {raw_csv_text[:200]}")
        return pd.DataFrame()
    try:
        # WSI API often has 2 header lines before the actual data headers
        df = pd.read_csv(StringIO(raw_csv_text), header=2)
        df.columns = df.columns.str.strip()
        return df
    except pd.errors.ParserError:
        print(f"  ⚠️ ParserError for {city_symbol}. Raw response snippet:\n{raw_csv_text[:500]}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print(f"  ⚠️ EmptyDataError after skipping headers for {city_symbol}. Raw response snippet:\n{raw_csv_text[:500]}")
        return pd.DataFrame()
    except Exception as e:
        print(f"  ⚠️ Unexpected error parsing CSV for {city_symbol}: {e}. Raw response snippet:\n{raw_csv_text[:500]}")
        return pd.DataFrame()

def fetch_wsi_historical_data(city_symbol, start_date_str_api, end_date_str_api):
    """
    Fetches historical daily observed data (Temps, HDD, CDD) for a single city.
    Makes two API calls: one for temps, one for degree days.
    Dates should be in MM/DD/YYYY format for the API.
    """
    print(f"  Fetching WSI data for {city_symbol} from {start_date_str_api} to {end_date_str_api}...")
    
    # Call 1: Fetch Temperatures
    params_temp = {
        "Account": USERNAME, "Profile": PROFILE, "Password": PASSWORD,
        "CityIds[]": city_symbol, "StartDate": start_date_str_api, "EndDate": end_date_str_api,
        "HistoricalProductId": "HISTORICAL_DAILY_OBSERVED", "TempUnits": "F",
        "IsTemp": "true", "IsDaily": "true", "IsDisplayDates": "false"
    }
    df_temp = pd.DataFrame()
    try:
        response_temp = requests.get(f"{BASE_SERVICE_URL}/GetHistoricalObservations", params=params_temp, timeout=60)
        response_temp.raise_for_status()
        df_temp_raw = parse_api_response_csv(response_temp.text, city_symbol)

        if not df_temp_raw.empty:
            rename_map_temp = {}
            for col in df_temp_raw.columns:
                col_s = str(col).strip().lower()
                if col_s == "date": rename_map_temp[col] = "Date"
                elif col_s == "min (f)": rename_map_temp[col] = "Min Temp"
                elif col_s == "max (f)": rename_map_temp[col] = "Max Temp"
                elif col_s == "avg (f)": rename_map_temp[col] = "Avg Temp"
            df_temp_raw.rename(columns=rename_map_temp, inplace=True)

            if "Date" in df_temp_raw.columns:
                df_temp_raw["Date"] = pd.to_datetime(df_temp_raw["Date"], format='%d-%b-%Y', errors='coerce')
                df_temp = df_temp_raw[["Date", "Min Temp", "Max Temp", "Avg Temp"]].copy()
                for col in ["Min Temp", "Max Temp", "Avg Temp"]:
                    if col in df_temp.columns:
                        df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
                    else:
                        df_temp[col] = np.nan # Add if missing
                # Calculate Avg Temp if not directly available but Min/Max are
                if "Avg Temp" not in df_temp.columns or df_temp["Avg Temp"].isnull().all():
                    if "Min Temp" in df_temp.columns and "Max Temp" in df_temp.columns:
                         # Ensure Min Temp and Max Temp are numeric before averaging
                        min_t = pd.to_numeric(df_temp["Min Temp"], errors='coerce')
                        max_t = pd.to_numeric(df_temp["Max Temp"], errors='coerce')
                        df_temp["Avg Temp"] = (min_t + max_t) / 2
            else:
                print(f"  ⚠️ 'Date' column not found in TEMP API response for {city_symbol}.")
    except requests.exceptions.RequestException as e:
        print(f"  ❌ ERROR fetching TEMP data for {city_symbol}: {e}")
    except Exception as e:
        print(f"  ❌ Unexpected error processing TEMP API data for {city_symbol}: {e}")

    # Call 2: Fetch HDD/CDD
    params_dd = {
        "Account": USERNAME, "Profile": PROFILE, "Password": PASSWORD,
        "CityIds[]": city_symbol, "StartDate": start_date_str_api, "EndDate": end_date_str_api,
        "HistoricalProductId": "HISTORICAL_DAILY_OBSERVED", "TempUnits": "F", 
        "IsTemp": "false", "IsDaily": "true", "IsDisplayDates": "false"
    }
    df_dd = pd.DataFrame()
    try:
        response_dd = requests.get(f"{BASE_SERVICE_URL}/GetHistoricalObservations", params=params_dd, timeout=60)
        response_dd.raise_for_status()
        df_dd_raw = parse_api_response_csv(response_dd.text, city_symbol)
        
        if not df_dd_raw.empty:
            rename_map_dd = {}
            for col in df_dd_raw.columns:
                col_s = str(col).strip().lower()
                if col_s == "date": rename_map_dd[col] = "Date"
                elif col_s == "hdd": rename_map_dd[col] = "HDD"
                elif col_s == "cdd": rename_map_dd[col] = "CDD"
            df_dd_raw.rename(columns=rename_map_dd, inplace=True)

            if "Date" in df_dd_raw.columns:
                df_dd_raw["Date"] = pd.to_datetime(df_dd_raw["Date"], format='%d-%b-%Y', errors='coerce')
                df_dd = df_dd_raw[["Date", "HDD", "CDD"]].copy()
                for col in ["HDD", "CDD"]:
                    if col in df_dd.columns:
                         df_dd[col] = pd.to_numeric(df_dd[col], errors='coerce')
                    else:
                        df_dd[col] = np.nan # Add if missing
            else:
                print(f"  ⚠️ 'Date' column not found in HDD/CDD API response for {city_symbol}.")
    except requests.exceptions.RequestException as e:
        print(f"  ❌ ERROR fetching HDD/CDD data for {city_symbol}: {e}")
    except Exception as e:
        print(f"  ❌ Unexpected error processing HDD/CDD API data for {city_symbol}: {e}")

    # Merge results
    if df_temp.empty and df_dd.empty:
        return pd.DataFrame()
    
    # Ensure 'Date' column exists and is datetime for merging
    if not df_temp.empty and 'Date' in df_temp.columns:
        df_temp['Date'] = pd.to_datetime(df_temp['Date'], errors='coerce')
        df_temp.dropna(subset=['Date'], inplace=True)
    if not df_dd.empty and 'Date' in df_dd.columns:
        df_dd['Date'] = pd.to_datetime(df_dd['Date'], errors='coerce')
        df_dd.dropna(subset=['Date'], inplace=True)

    if df_temp.empty:
        return df_dd
    elif df_dd.empty:
        return df_temp
    else:
        merged_df = pd.merge(df_temp, df_dd, on="Date", how="outer")
        return merged_df


def calculate_10yr_normals(df, effective_start_date_for_recalc):
    """
    Calculates 10-year normals for dates on or after effective_start_date_for_recalc.
    Older dates preserve their existing 10-year normal values.
    Historical lookups use the entire df for context.
    """
    print(f"\nCalculating 10-year normals for dates on or after {effective_start_date_for_recalc.strftime('%Y-%m-%d')}...")
    if df.empty or 'Date' not in df.columns:
        print("  DataFrame is empty or 'Date' column missing. Skipping 10-year normal calculation.")
        for col in TEN_YEAR_NORMAL_TARGET_COLS: # Ensure target columns exist
            if col not in df.columns: df[col] = np.nan
        return df

    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Full dataset copy for historical lookups, with Year/Month/Day helpers
    df_historical_context = df.copy()
    df_historical_context['Year'] = df_historical_context['Date'].dt.year
    df_historical_context['Month'] = df_historical_context['Date'].dt.month
    df_historical_context['Day'] = df_historical_context['Date'].dt.day

    # Split data: one part to preserve, one part to recalculate normals for
    df_to_preserve_normals = df[df['Date'] < effective_start_date_for_recalc].copy()
    df_for_recalc_normals = df[df['Date'] >= effective_start_date_for_recalc].copy()

    if df_for_recalc_normals.empty:
        print("  No recent data to recalculate 10-year normals for. Preserving all existing normals.")
        return df # Original df already has the structure

    print(f"  Recalculating normals for {len(df_for_recalc_normals)} rows.")
    # Add Year/Month/Day to the part being recalculated for iteration
    df_for_recalc_normals['Year'] = df_for_recalc_normals['Date'].dt.year
    df_for_recalc_normals['Month'] = df_for_recalc_normals['Date'].dt.month
    df_for_recalc_normals['Day'] = df_for_recalc_normals['Date'].dt.day
    
    # Initialize target normal columns with NaN in the df_for_recalc_normals part
    for target_col in TEN_YEAR_NORMAL_TARGET_COLS:
        df_for_recalc_normals[target_col] = np.nan

    # Store calculated normals for the recalc part
    calculated_normals_list = []

    for (city, year, month, day), group in df_for_recalc_normals.groupby(['City Symbol', 'Year', 'Month', 'Day']):
        current_date_in_group = group['Date'].iloc[0]
        
        start_year_window = year - 10
        end_year_window = year - 1
        
        # Historical data lookup uses df_historical_context (the full dataset's context)
        historical_period_data = df_historical_context[
            (df_historical_context['City Symbol'] == city) &
            (df_historical_context['Month'] == month) &
            (df_historical_context['Day'] == day) &
            (df_historical_context['Year'] >= start_year_window) &
            (df_historical_context['Year'] <= end_year_window)
        ]
        
        # Create a dictionary for the current row's calculated normals
        current_row_normals = {'City Symbol': city, 'Date': current_date_in_group}
        if not historical_period_data.empty:
            for i, source_col in enumerate(TEN_YEAR_NORMAL_SOURCE_COLS):
                target_col = TEN_YEAR_NORMAL_TARGET_COLS[i]
                if source_col in historical_period_data.columns:
                    mean_val = historical_period_data[source_col].mean(skipna=True)
                    current_row_normals[target_col] = mean_val if pd.notna(mean_val) else np.nan
                else:
                    current_row_normals[target_col] = np.nan # Source column not in history
        else: # No historical data for this specific day in the 10yr window
            for target_col in TEN_YEAR_NORMAL_TARGET_COLS:
                current_row_normals[target_col] = np.nan
        calculated_normals_list.append(current_row_normals)

    if calculated_normals_list:
        # Create a DataFrame from the list of calculated normals
        new_normals_df = pd.DataFrame(calculated_normals_list)
        
        # Merge these new normals back into df_for_recalc_normals
        # First, drop the placeholder NaN normal columns from df_for_recalc_normals
        df_for_recalc_normals = df_for_recalc_normals.drop(columns=TEN_YEAR_NORMAL_TARGET_COLS, errors='ignore')
        # Merge (ensure Date is consistent type if issues arise)
        new_normals_df['Date'] = pd.to_datetime(new_normals_df['Date'])
        df_for_recalc_normals['Date'] = pd.to_datetime(df_for_recalc_normals['Date'])
        df_for_recalc_normals = pd.merge(df_for_recalc_normals, new_normals_df, on=['City Symbol', 'Date'], how='left')
    
    # Drop helper columns from df_for_recalc_normals
    df_for_recalc_normals = df_for_recalc_normals.drop(columns=['Year', 'Month', 'Day'], errors='ignore')
        
    # Concatenate the part with preserved normals and the part with recalculated normals
    final_df = pd.concat([df_to_preserve_normals, df_for_recalc_normals], ignore_index=True)
    
    print("Finished calculating targeted 10-year normals.")
    return final_df


def find_missing_dates_for_df(df, group_col='City Symbol'):
    """ Identifies and prints missing dates for each group in the DataFrame. """
    if 'Date' not in df.columns:
        print("⚠️ 'Date' column not found. Cannot check for missing dates.")
        return

    df['Date'] = pd.to_datetime(df['Date'])
    print("\n--- Checking for Missing Dates in Updated Data ---")
    for group_name, group_df in df.groupby(group_col):
        if group_df.empty:
            continue
        
        min_date = group_df['Date'].min()
        max_date = group_df['Date'].max()
        
        if pd.isna(min_date) or pd.isna(max_date):
            print(f"  ⚠️ Could not determine date range for {group_name} (min/max date is NaT).")
            continue

        expected_dates = pd.date_range(start=min_date, end=max_date)
        missing_dates_mask = ~expected_dates.isin(group_df['Date'])
        missing_dates = expected_dates[missing_dates_mask]
        
        if not missing_dates.empty:
            print(f"  Missing dates for {group_name} (showing up to 5):")
            for i, date in enumerate(missing_dates):
                if i < 5:
                    print(f"    {date.strftime('%Y-%m-%d')}")
                else:
                    print(f"    ... and {len(missing_dates) - 5} more.")
                    break
        else:
            print(f"  No missing dates found for {group_name} between {min_date.strftime('%Y-%m-%d')} and {max_date.strftime('%Y-%m-%d')}.")


# --- Main Script Logic ---
def update_weather_file():
    if not all([USERNAME, PROFILE, PASSWORD]):
        print("❌ Critical Error: WSI credentials not found. Check .env: WSI_ACCOUNT_USERNAME, WSI_PROFILE_EMAIL, WSI_PASSWORD.")
        return

    print(f"Attempting to load WEATHER.csv from: {weather_csv_path}")
    if not os.path.exists(weather_csv_path):
        print(f"❌ ERROR: File not found: {weather_csv_path}. Cannot proceed without initial city list.")
        return

    try:
        weather_df = pd.read_csv(weather_csv_path, low_memory=False)
    except Exception as e:
        print(f"❌ Error loading WEATHER.csv: {e}")
        return

    weather_df.columns = weather_df.columns.str.strip()
    cols_to_keep_initial = [col for col in EXPECTED_COLUMNS if col in weather_df.columns]
    weather_df = weather_df[cols_to_keep_initial].copy()

    if 'Date' in weather_df.columns:
        weather_df['Date'] = pd.to_datetime(weather_df['Date'], errors='coerce')
        weather_df.dropna(subset=['Date'], inplace=True)
    else:
        print("❌ 'Date' column not found in WEATHER.csv. Cannot proceed.")
        return

    if 'City Symbol' not in weather_df.columns:
        print("❌ 'City Symbol' column not found in WEATHER.csv. Cannot proceed.")
        return
        
    unique_city_symbols = weather_df['City Symbol'].unique()
    if len(unique_city_symbols) == 0:
        print("No city symbols found in WEATHER.csv. Nothing to update.")
        return
        
    city_title_map = {}
    if 'City Title' in weather_df.columns:
        city_title_map = weather_df[['City Symbol', 'City Title']].drop_duplicates().set_index('City Symbol')['City Title'].to_dict()

    print(f"Found {len(unique_city_symbols)} unique city symbols for processing.")

    last_date_in_file = weather_df['Date'].max() if not weather_df.empty else None
    today = datetime.today()
    
    # Define fetch_start_date carefully
    if last_date_in_file:
        # This is the start of the window for API fetching AND for 10yr normal recalc
        fetch_start_date = last_date_in_file - timedelta(days=30) 
        # If the file is very up-to-date, ensure we don't set fetch_start_date to be in the future
        # relative to today, or even today if we want at least one day of refresh.
        if fetch_start_date.date() >= today.date():
             fetch_start_date = today - timedelta(days=1) 
             print(f"Last date in file is very recent ({last_date_in_file.strftime('%Y-%m-%d')}). Setting fetch/recalc start to {fetch_start_date.strftime('%Y-%m-%d')}.")
        else:
             print(f"Last date in file: {last_date_in_file.strftime('%Y-%m-%d')}. API fetch & 10yr normal recalc will start from {fetch_start_date.strftime('%Y-%m-%d')}.")
    else: 
        # If file is empty or no dates, fetch last year, and all fetched data will have normals calculated
        fetch_start_date = today - timedelta(days=365) 
        print(f"No valid last date in file. Will attempt to fetch data and calculate normals from {fetch_start_date.strftime('%Y-%m-%d')}.")

    fetch_end_date = today 
    
    fetch_start_date_api_str = fetch_start_date.strftime("%m/%d/%Y")
    fetch_end_date_api_str = fetch_end_date.strftime("%m/%d/%Y")

    all_new_city_data = []
    if fetch_start_date.date() <= fetch_end_date.date():
        for city_sym in unique_city_symbols:
            city_title = city_title_map.get(city_sym, city_sym) 
            new_data = fetch_wsi_historical_data(city_sym, fetch_start_date_api_str, fetch_end_date_api_str)
            if not new_data.empty:
                new_data['City Symbol'] = city_sym
                new_data['City Title'] = city_title
                all_new_city_data.append(new_data)
    else:
        print("\nFetch start date is after fetch end date. No new data will be fetched.")

    if not all_new_city_data:
        print("\nNo new data fetched from API for any city.")
        # If no new data, we still want to pass the original df to calculate_10yr_normals
        # if there's a need to ensure normals are calculated for a recent period
        # (e.g. if script failed previously before normal calc but after API fetch).
        # However, if fetch_start_date was determined from an empty file, this logic is complex.
        # For simplicity, if no new data, we assume the 10yr normal calc will operate on existing data.
        updated_df_before_normals = weather_df.copy()
    else:
        newly_fetched_df = pd.concat(all_new_city_data, ignore_index=True)
        print(f"\nFetched {len(newly_fetched_df)} new/updated rows in total from API.")

        condition_to_remove = (weather_df['City Symbol'].isin(newly_fetched_df['City Symbol'].unique())) & \
                              (weather_df['Date'] >= pd.to_datetime(fetch_start_date)) # ensure fetch_start_date is datetime for comparison
        
        weather_df_filtered = weather_df[~condition_to_remove]
        updated_df_before_normals = pd.concat([weather_df_filtered, newly_fetched_df], ignore_index=True)

    updated_df_before_normals.sort_values(by=['City Symbol', 'Date'], inplace=True)
    updated_df_before_normals.drop_duplicates(subset=['City Symbol', 'Date'], keep='last', inplace=True)

    # Calculate 10-year normals, targeting recalculation from fetch_start_date
    updated_df_with_normals = calculate_10yr_normals(updated_df_before_normals, fetch_start_date)

    # Ensure final columns and order
    final_df_cols = []
    for col in EXPECTED_COLUMNS:
        if col not in updated_df_with_normals.columns:
            updated_df_with_normals[col] = np.nan 
        final_df_cols.append(col)
    final_df = updated_df_with_normals[final_df_cols] # Enforce order and selection

    print(f"\nTotal rows in final DataFrame: {len(final_df)}")

    find_missing_dates_for_df(final_df.copy())

    try:
        if 'Date' in final_df.columns: # Ensure Date is datetime before formatting
             final_df['Date'] = pd.to_datetime(final_df['Date']).dt.strftime('%Y-%m-%d')
        final_df.to_csv(weather_csv_path, index=False)
        print(f"\n✅ Successfully saved updated data to: {weather_csv_path}")
    except Exception as e:
        print(f"\n❌ Error saving updated data to CSV: {e}")

if __name__ == "__main__":
    update_weather_file()