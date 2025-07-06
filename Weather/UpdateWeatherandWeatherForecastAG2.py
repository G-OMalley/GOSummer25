"""
================================================================================================
UpdateWeatherandWeatherForecastAG2.py
================================================================================================
Description:
This script automates the process of fetching, processing, and saving historical and forecast
weather data from the WSI Trader API. It is designed to be the single source for maintaining
two key data files: WEATHER.csv (historical observations) and WEATHERforecast.csv (future
predictions).

Key Features:
- Master City List: Retrieves the list of cities to process from a central 'PriceAdmin.csv' file.
- Smart Historical Updates: Checks the existing WEATHER.csv, removes the last 14 days of data
  to ensure data quality, and only fetches data from that point forward. This prevents
  re-downloading the entire historical dataset on each run.
- Comprehensive Data Fetching: Gathers both daily and hourly observations to create a rich
  dataset, including temperature, precipitation, dewpoint, cloud cover, wind speed, and
  "feels like" temperatures.
- 10-Year Normals Calculation: For both historical and forecast data, it calculates the
  10-year average for key metrics on a given calendar day, providing crucial context.
- Custom Forecast Adjustments: Applies a specific logic to adjust raw forecast temperatures,
  refining the predictive data.
- Dual File Output: Produces two distinct, consistently formatted CSV files for clear
  separation of historical and forecast data.

Execution Flow:
1. Load configuration and credentials.
2. Get the list of city symbols from PriceAdmin.csv.
3. Update WEATHER.csv with the latest historical data.
4. Update WEATHERforecast.csv using the newly updated historical data for context.
5. Print progress and confirmation messages throughout the process.

================================================================================================
DISCLAIMER AND LIMITATION OF LIABILITY
================================================================================================
This script is provided "as is" without warranty of any kind, either expressed or implied.
The user assumes full responsibility for the use of this script and any results or
consequences that may arise. The author or distributor shall not be liable for any damages
arising from the use of this script. By using this script, you agree to these terms.
"""

import pandas as pd
import requests
from io import StringIO
from datetime import datetime, timedelta
import os
import numpy as np
from dotenv import load_dotenv
import time

# --- Configuration & Setup ---
print("--- Initializing Weather Data Update Script ---")

# ============================================================================================
# ONE-TIME BACKFILL CONFIGURATION
# ============================================================================================
# Set this to True to run a one-time historical backfill. This will fetch and process data
# for ALL historical records from 2015-01-01 to add the new columns.
# WARNING: This will take a very long time to run.
# After the backfill is complete, SET THIS BACK TO False for normal, fast daily updates.
PERFORM_FULL_BACKFILL = False
# ============================================================================================


# Load credentials from .env file in the script's directory
# Ensure your .env file has: WSI_ACCOUNT_USERNAME, WSI_PROFILE_EMAIL, WSI_PASSWORD
load_dotenv()
USERNAME = os.getenv("WSI_ACCOUNT_USERNAME")
PROFILE = os.getenv("WSI_PROFILE_EMAIL")
PASSWORD = os.getenv("WSI_PASSWORD")

# --- Define File Paths ---
# The script assumes it is located in the 'Weather' folder.
# The 'INFO' folder is at the same level as the 'Weather' folder's parent.
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_project_dir = os.path.dirname(script_dir) # Navigates up to 'TraderHelper'
    INFO_DIR = os.path.join(base_project_dir, "INFO")

    PRICE_ADMIN_PATH = os.path.join(INFO_DIR, "PriceAdmin.csv")
    WEATHER_CSV_PATH = os.path.join(INFO_DIR, "WEATHER.csv")
    WEATHER_FORECAST_CSV_PATH = os.path.join(INFO_DIR, "WEATHERforecast.csv")
    print(f"INFO directory set to: {INFO_DIR}")
except Exception:
    # Fallback for environments where __file__ might not be defined
    print("Warning: Could not determine script path automatically. Using hardcoded paths.")
    _user_home = os.path.expanduser("~")
    INFO_DIR = os.path.join(_user_home, "OneDrive", "Desktop", "Coding", "TraderHelper", "INFO")
    PRICE_ADMIN_PATH = os.path.join(INFO_DIR, "PriceAdmin.csv")
    WEATHER_CSV_PATH = os.path.join(INFO_DIR, "WEATHER.csv")
    WEATHER_FORECAST_CSV_PATH = os.path.join(INFO_DIR, "WEATHERforecast.csv")
    print(f"INFO directory fall-back: {INFO_DIR}")


# --- API Configuration ---
BASE_SERVICE_URL = "https://www.wsitrader.com/Services/CSVDownloadService.svc"

# --- Define Final Column Structure ---
# This order will be enforced on both output files.
CORE_COLUMNS = [
    'City Symbol', 'City Title', 'Date', 'Min Temp', 'Max Temp', 'Avg Temp', 'CDD', 'HDD'
]
NORMAL_COLUMNS = [
    '10yr Min Temp', '10yr Max Temp', '10yr Avg Temp', '10yr CDD', '10yr HDD'
]
NEW_ATTRIBUTES = [
    'Max Dewpoint', 'Avg Cloud Cover', 'Max Surface Wind',
    'Min Feels Like', 'Max Feels Like', 'Daily Precip Amount'
]
FINAL_COLUMN_ORDER = CORE_COLUMNS + NORMAL_COLUMNS + NEW_ATTRIBUTES

# --- Helper Functions ---

def get_city_symbols_from_priceadmin(file_path):
    """
    Reads the PriceAdmin.csv file and extracts a unique list of city symbols
    from Column J (index 9).
    """
    print(f"\n--- Reading City List from {os.path.basename(file_path)} ---")
    try:
        if not os.path.exists(file_path):
            print(f"❌ CRITICAL ERROR: PriceAdmin.csv not found at path: {file_path}")
            return None

        # Read only the 10th column (Column J)
        df = pd.read_csv(file_path, usecols=[9], header=0)
        # The column name is the first row of the file, get it dynamically
        city_symbol_col_name = df.columns[0]

        city_symbols = df[city_symbol_col_name].dropna().unique().tolist()
        
        if not city_symbols:
            print("❌ CRITICAL ERROR: No city symbols found in Column J of PriceAdmin.csv.")
            return None
            
        print(f"✅ Found {len(city_symbols)} unique city symbols to process.")
        return city_symbols
    except Exception as e:
        print(f"❌ CRITICAL ERROR reading PriceAdmin.csv: {e}")
        return None

def calculate_feels_like(temp, heat_index, wind_chill, wind_speed):
    """
    Determines the 'feels like' temperature based on conditions.
    This function is vectorized for performance with pandas.
    """
    conditions = [
        (temp > 80),
        (temp < 50) & (wind_speed >= 3)
    ]
    choices = [heat_index, wind_chill]
    return np.select(conditions, choices, default=temp)

def fetch_city_titles(city_symbols_list):
    """Fetches full city names/titles for the given symbols."""
    print("🔄 Fetching city titles for symbols...")
    url = f"{BASE_SERVICE_URL}/GetCityIds"
    params = {"Account": USERNAME, "Profile": PROFILE, "Password": PASSWORD}
    try:
        response = requests.get(url, params=params, timeout=45)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        df.columns = df.columns.str.strip()
        
        if "SiteId" not in df.columns or "Station Name" not in df.columns:
            print("❌ ERROR: 'SiteId' or 'Station Name' not found in GetCityIds response.")
            return {symbol: symbol for symbol in city_symbols_list} # Fallback

        df = df[df['SiteId'].isin(city_symbols_list)]
        city_title_map = pd.Series(df['Station Name'].values, index=df['SiteId']).to_dict()
        
        # Add any missing symbols with the symbol as the title
        for symbol in city_symbols_list:
            if symbol not in city_title_map:
                city_title_map[symbol] = symbol
        
        print("✅ Successfully fetched city titles.")
        return city_title_map
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch city titles due to an error: {e}. Using symbols as titles.")
        return {symbol: symbol for symbol in city_symbols_list}


def calculate_10yr_normals(df_to_update, historical_context_df):
    """
    Calculates 10-year normals for a given DataFrame using a historical context.
    """
    print("🔄 Calculating 10-year normals...")
    if historical_context_df.empty:
        print("⚠️ Historical context is empty. Cannot calculate 10-year normals.")
        for col in NORMAL_COLUMNS:
            df_to_update[col] = np.nan
        return df_to_update

    hist_df = historical_context_df.copy()
    hist_df['Date'] = pd.to_datetime(hist_df['Date'])
    hist_df['MonthDay'] = hist_df['Date'].dt.strftime('%m-%d')
    hist_df['Year'] = hist_df['Date'].dt.year

    # Columns for which to calculate normals
    source_cols = ['Min Temp', 'Max Temp', 'Avg Temp', 'CDD', 'HDD']
    
    all_normals = []

    # Create a unique list of city-date combinations to calculate normals for
    dates_to_calc = df_to_update[['City Symbol', 'Date']].drop_duplicates()
    
    for index, row in dates_to_calc.iterrows():
        city_symbol = row['City Symbol']
        current_date = pd.to_datetime(row['Date'])
        month_day_str = current_date.strftime('%m-%d')
        current_year = current_date.year
        
        # Find data for the same day in the previous 10 years
        city_historical_data = hist_df[
            (hist_df['City Symbol'] == city_symbol) &
            (hist_df['MonthDay'] == month_day_str) &
            (hist_df['Year'] >= current_year - 10) &
            (hist_df['Year'] < current_year)
        ]
        
        normal_values = {'City Symbol': city_symbol, 'Date': current_date}
        for i, source_col in enumerate(source_cols):
            target_col = NORMAL_COLUMNS[i]
            if not city_historical_data.empty and source_col in city_historical_data.columns:
                mean_val = pd.to_numeric(city_historical_data[source_col], errors='coerce').mean()
                normal_values[target_col] = mean_val
            else:
                normal_values[target_col] = np.nan
        all_normals.append(normal_values)
    
    if not all_normals:
        print("⚠️ No normals were calculated.")
        for col in NORMAL_COLUMNS:
            df_to_update[col] = np.nan
        return df_to_update

    normals_df = pd.DataFrame(all_normals)
    normals_df['Date'] = pd.to_datetime(normals_df['Date'])
    
    # Merge the calculated normals back into the dataframe
    df_to_update['Date'] = pd.to_datetime(df_to_update['Date'])
    # Drop old normal columns if they exist to prevent merge conflicts
    cols_to_drop = [col for col in NORMAL_COLUMNS if col in df_to_update.columns]
    if cols_to_drop:
        df_to_update = df_to_update.drop(columns=cols_to_drop)

    updated_df = pd.merge(df_to_update, normals_df, on=['City Symbol', 'Date'], how='left')
    print("✅ 10-year normals calculation complete.")
    return updated_df

# --- Main Processing Functions ---

def update_historical_data(city_symbols, city_titles):
    """
    Fetches and updates the historical WEATHER.csv file.
    """
    print("\n" + "="*50)
    print("--- Starting Historical Data Update (WEATHER.csv) ---")
    print("="*50)

    # 1. Determine the fetch start date
    fetch_start_date_str = '2015-01-01'
    existing_weather_df = pd.DataFrame()

    if PERFORM_FULL_BACKFILL:
        print("‼️ PERFORMING FULL HISTORICAL BACKFILL. This will take a long time.")
        # When backfilling, we start from the beginning and don't need the existing file.
        existing_weather_df = pd.DataFrame()
    elif os.path.exists(WEATHER_CSV_PATH):
        try:
            print(f"Reading existing data from {os.path.basename(WEATHER_CSV_PATH)} for standard update...")
            existing_weather_df = pd.read_csv(WEATHER_CSV_PATH)
            if not existing_weather_df.empty and 'Date' in existing_weather_df.columns:
                existing_weather_df['Date'] = pd.to_datetime(existing_weather_df['Date'])
                last_date = existing_weather_df['Date'].max()
                # Go back 14 days from the last record
                fetch_start_date = last_date - timedelta(days=14)
                fetch_start_date_str = fetch_start_date.strftime('%Y-%m-%d')
                print(f"Last record date: {last_date.date()}. Setting fetch start date to {fetch_start_date.date()}.")
            else:
                 print("WEATHER.csv is empty or has no 'Date' column. Will fetch from 2015-01-01.")
        except Exception as e:
            print(f"⚠️ Warning: Could not read WEATHER.csv ({e}). Will fetch from 2015-01-01.")
    else:
        print("WEATHER.csv not found. Will create a new file and fetch from 2015-01-01.")

    end_date_str = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d') # Yesterday
    
    # API date format is MM/DD/YYYY
    start_date_api = datetime.strptime(fetch_start_date_str, "%Y-%m-%d").strftime("%m/%d/%Y")
    end_date_api = datetime.strptime(end_date_str, "%Y-%m-%d").strftime("%m/%d/%Y")
    
    # 2. Fetch new data for all cities
    all_new_city_data = []
    for i, city_id in enumerate(city_symbols):
        print(f"\n🔄 ({i+1}/{len(city_symbols)}) Fetching historical data for: {city_titles.get(city_id, city_id)} ({city_id})")
        
        # --- Get Observed Daily Data ---
        daily_df = pd.DataFrame()
        try:
            daily_url = (f"{BASE_SERVICE_URL}/GetHistoricalObservations?Account={USERNAME}&profile={PROFILE}&password={PASSWORD}"
                         f"&TempUnits=F&HistoricalProductID=HISTORICAL_DAILY_OBSERVED&StartDate={start_date_api}&EndDate={end_date_api}"
                         f"&IsDisplayDates=false&IsTemp=true&CityIds[]={city_id}")
            daily_response = requests.get(daily_url, timeout=60)
            daily_response.raise_for_status()
            if "no data" not in daily_response.text.lower():
                daily_df = pd.read_csv(StringIO(daily_response.text), skiprows=3, header=None, names=["Date", "Min Temp", "Max Temp", "AvgTemp", "Daily Precip Amount"])
                daily_df["Date"] = pd.to_datetime(daily_df["Date"], format='%d-%b-%Y').dt.date
            else:
                print(f"  - No daily data returned for {city_id}.")
        except Exception as e:
            print(f"  - ❌ ERROR fetching daily data for {city_id}: {e}")

        # --- Get Hourly Observed Data ---
        hourly_summary = pd.DataFrame()
        try:
            hourly_url = (f"{BASE_SERVICE_URL}/GetHistoricalObservations?Account={USERNAME}&Profile={PROFILE}&Password={PASSWORD}"
                          f"&HistoricalProductID=HISTORICAL_HOURLY_OBSERVED&DataTypes[]=dewpoint&DataTypes[]=temperature"
                          f"&DataTypes[]=cloudCover&DataTypes[]=windSpeed&DataTypes[]=heatIndex&DataTypes[]=windChill"
                          f"&TempUnits=F&StartDate={start_date_api}&EndDate={end_date_api}&CityIds[]={city_id}")
            hourly_response = requests.get(hourly_url, timeout=120)
            hourly_response.raise_for_status()
            
            if "no data" not in hourly_response.text.lower():
                hourly_df = pd.read_csv(StringIO(hourly_response.text), skiprows=2, header=None, 
                                        names=["Date", "Hour", "Temperature", "Dewpoint", "WindChill", "HeatIndex", "WindSpeed", "CloudCover", "Precip_hourly"])
                
                for col in ["Temperature", "Dewpoint", "WindChill", "HeatIndex", "WindSpeed"]:
                    hourly_df[col] = pd.to_numeric(hourly_df[col], errors='coerce')
                hourly_df["CloudCover"] = hourly_df["CloudCover"].astype(str).str.replace("%", "", regex=False)
                hourly_df["CloudCover"] = pd.to_numeric(hourly_df["CloudCover"], errors='coerce')

                hourly_df["FeelsLike"] = calculate_feels_like(hourly_df["Temperature"], hourly_df["HeatIndex"], hourly_df["WindChill"], hourly_df["WindSpeed"])
                hourly_df["Date"] = pd.to_datetime(hourly_df["Date"], format='%m/%d/%Y').dt.date

                hourly_summary = hourly_df.groupby("Date").agg(
                    Max_Dewpoint=('Dewpoint', 'max'),
                    Avg_Cloud_Cover=('CloudCover', 'mean'),
                    Max_Surface_Wind=('WindSpeed', 'max'),
                    Min_Feels_Like=('FeelsLike', 'min'),
                    Max_Feels_Like=('FeelsLike', 'max')
                ).reset_index()
                hourly_summary.columns = ["Date", "Max Dewpoint", "Avg Cloud Cover", "Max Surface Wind", "Min Feels Like", "Max Feels Like"]
            else:
                print(f"  - No hourly data returned for {city_id}.")
        except Exception as e:
            print(f"  - ❌ ERROR fetching hourly data for {city_id}: {e}")

        # --- Combine and process for the city ---
        if not daily_df.empty:
            combined_df = daily_df
            if not hourly_summary.empty:
                combined_df = pd.merge(daily_df, hourly_summary, on="Date", how="left")
            
            combined_df.insert(0, "City Symbol", city_id)
            combined_df.insert(1, "City Title", city_titles.get(city_id, city_id))
            
            # Calculate derived columns
            combined_df['Avg Temp'] = (pd.to_numeric(combined_df['Min Temp'], errors='coerce') + pd.to_numeric(combined_df['Max Temp'], errors='coerce')) / 2
            combined_df['HDD'] = (65 - combined_df['Avg Temp']).clip(lower=0)
            combined_df['CDD'] = (combined_df['Avg Temp'] - 65).clip(lower=0)

            all_new_city_data.append(combined_df)
            print(f"  - ✅ Successfully processed data for {city_id}.")
        else:
            print(f"  - ⚠️ Skipping {city_id} due to lack of daily data.")
        
        time.sleep(0.5) # Small delay to be courteous to the API

    if not all_new_city_data:
        print("\n❌ No new historical data was fetched for any city. Aborting historical update.")
        return

    # 3. Combine new data and merge with existing data
    print("\n🔄 Merging new data with existing historical records...")
    newly_fetched_df = pd.concat(all_new_city_data, ignore_index=True)
    newly_fetched_df['Date'] = pd.to_datetime(newly_fetched_df['Date'])
    
    # Filter out the last 14 days from the old data if not doing a full backfill
    if not PERFORM_FULL_BACKFILL and not existing_weather_df.empty:
        fetch_start_dt = datetime.strptime(fetch_start_date_str, '%Y-%m-%d')
        existing_weather_df = existing_weather_df[existing_weather_df['Date'] < fetch_start_dt]
        
        # Combine old (filtered) and new data
        final_historical_df = pd.concat([existing_weather_df, newly_fetched_df], ignore_index=True)
    else:
        # This path is taken for a full backfill or if the original file was empty
        final_historical_df = newly_fetched_df

    # Remove duplicates, keeping the newest record for any given city/date
    final_historical_df = final_historical_df.sort_values(by=['City Symbol', 'Date']).drop_duplicates(subset=['City Symbol', 'Date'], keep='last')
    
    # 4. Calculate 10-year normals
    final_historical_df = calculate_10yr_normals(final_historical_df, final_historical_df)

    # 5. Format and save
    # Ensure all final columns exist, fill with NaN if not
    for col in FINAL_COLUMN_ORDER:
        if col not in final_historical_df.columns:
            final_historical_df[col] = np.nan

    final_historical_df = final_historical_df[FINAL_COLUMN_ORDER]
    final_historical_df['Date'] = pd.to_datetime(final_historical_df['Date']).dt.strftime('%Y-%m-%d')
    
    try:
        final_historical_df.to_csv(WEATHER_CSV_PATH, index=False)
        print(f"\n✅✅✅ Successfully updated and saved historical data to: {WEATHER_CSV_PATH}")
        print(f"Total historical rows: {len(final_historical_df)}")
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR saving updated WEATHER.csv: {e}")


def update_forecast_data(city_symbols, city_titles):
    """
    Fetches and creates the WEATHERforecast.csv file.
    """
    print("\n" + "="*50)
    print("--- Starting Forecast Data Update (WEATHERforecast.csv) ---")
    print("="*50)

    # 1. Load the brand new historical data for context
    try:
        historical_context_df = pd.read_csv(WEATHER_CSV_PATH)
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Cannot read WEATHER.csv for forecast context: {e}")
        print("Aborting forecast update.")
        return

    all_forecast_data = []
    for i, city_id in enumerate(city_symbols):
        print(f"\n🔄 ({i+1}/{len(city_symbols)}) Fetching forecast data for: {city_titles.get(city_id, city_id)} ({city_id})")
        
        try:
            forecast_url = (f"{BASE_SERVICE_URL}/GetHourlyForecast?Account={USERNAME}&Profile={PROFILE}&Password={PASSWORD}"
                            f"&region=NA&SiteIds[]={city_id}&TempUnits=F")
            forecast_response = requests.get(forecast_url, timeout=120)
            forecast_response.raise_for_status()

            if "no data" in forecast_response.text.lower():
                print(f"  - No forecast data returned for {city_id}.")
                continue

            forecast_df = pd.read_csv(StringIO(forecast_response.text), skiprows=2, header=None,
                                      names=["DateHour", "Temperature", "TempDiff", "TempNormal", "DewPoint", "CloudCover",
                                             "FeelsLikeTemp", "FeelsLikeTempDiff", "Precip", "WindDirection", "WindSpeed", "GHI"])
            
            forecast_df["Date"] = pd.to_datetime(forecast_df["DateHour"], format="mixed").dt.date
            
            for col in ["Temperature", "DewPoint", "CloudCover", "FeelsLikeTemp", "Precip", "WindSpeed"]:
                forecast_df[col] = pd.to_numeric(forecast_df[col], errors="coerce")

            forecast_summary = forecast_df.groupby("Date").agg(
                Min_Temp=('Temperature', 'min'),
                Max_Temp=('Temperature', 'max'),
                Max_Dewpoint=('DewPoint', 'max'),
                Min_Feels_Like=('FeelsLikeTemp', 'min'),
                Max_Feels_Like=('FeelsLikeTemp', 'max'),
                Max_Surface_Wind=('WindSpeed', 'max'),
                Daily_Precip_Amount=('Precip', 'sum'),
                Avg_Cloud_Cover=('CloudCover', 'mean')
            ).reset_index()
            
            # Apply custom forecast adjustments
            forecast_summary['Max_Temp'] = forecast_summary.apply(
                lambda row: row['Max_Temp'] + (2 * (row['Max_Temp'] - row['Min_Temp']) / 18)
                if (row['Max_Temp'] - row['Min_Temp']) <= 18 else row['Max_Temp'] + 2,
                axis=1
            )
            forecast_summary['Min_Temp'] -= 1

            # Rename columns to match final structure
            forecast_summary.rename(columns={'Min_Temp': 'Min Temp', 'Max_Temp': 'Max Temp'}, inplace=True)
            
            # Calculate derived columns
            forecast_summary['Avg Temp'] = (forecast_summary['Min Temp'] + forecast_summary['Max Temp']) / 2
            forecast_summary['HDD'] = (65 - forecast_summary['Avg Temp']).clip(lower=0)
            forecast_summary['CDD'] = (forecast_summary['Avg Temp'] - 65).clip(lower=0)

            forecast_summary.insert(0, 'City Symbol', city_id)
            forecast_summary.insert(1, 'City Title', city_titles.get(city_id, city_id))
            
            all_forecast_data.append(forecast_summary)
            print(f"  - ✅ Successfully processed forecast for {city_id}.")

        except Exception as e:
            print(f"  - ❌ ERROR fetching/processing forecast for {city_id}: {e}")
        
        time.sleep(0.5) # Small delay

    if not all_forecast_data:
        print("\n❌ No forecast data was fetched for any city. Aborting forecast update.")
        return

    # 2. Combine all city forecasts
    final_forecast_df = pd.concat(all_forecast_data, ignore_index=True)

    # 3. Calculate 10-year normals using the historical context
    final_forecast_df = calculate_10yr_normals(final_forecast_df, historical_context_df)

    # 4. Format and save
    for col in FINAL_COLUMN_ORDER:
        if col not in final_forecast_df.columns:
            final_forecast_df[col] = np.nan
    
    final_forecast_df = final_forecast_df[FINAL_COLUMN_ORDER]
    final_forecast_df['Date'] = pd.to_datetime(final_forecast_df['Date']).dt.strftime('%Y-%m-%d')
    
    try:
        final_forecast_df.to_csv(WEATHER_FORECAST_CSV_PATH, index=False)
        print(f"\n✅✅✅ Successfully created and saved forecast data to: {WEATHER_FORECAST_CSV_PATH}")
        print(f"Total forecast rows: {len(final_forecast_df)}")
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR saving WEATHERforecast.csv: {e}")


def main():
    """
    Main function to orchestrate the entire weather data update process.
    """
    if not all([USERNAME, PROFILE, PASSWORD]):
        print("❌ CRITICAL ERROR: WSI credentials not found in .env file or environment variables.")
        print("Please ensure WSI_ACCOUNT_USERNAME, WSI_PROFILE_EMAIL, and WSI_PASSWORD are set.")
        return

    city_symbols = get_city_symbols_from_priceadmin(PRICE_ADMIN_PATH)
    if not city_symbols:
        return # Error message is handled inside the function
    
    city_titles = fetch_city_titles(city_symbols)

    update_historical_data(city_symbols, city_titles)
    
    # This function relies on the output of the historical update
    update_forecast_data(city_symbols, city_titles)
    
    print("\n--- Weather Data Update Script Finished ---")

if __name__ == "__main__":
    main()
