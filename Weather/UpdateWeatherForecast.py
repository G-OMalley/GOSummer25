import pandas as pd
import os
import requests
from io import StringIO
from dotenv import load_dotenv
from datetime import datetime, timedelta
import numpy as np
import re # For extracting city symbol

# --- Configuration & Constants ---
# Load credentials from .env file
load_dotenv()
USERNAME = os.getenv("WSI_ACCOUNT_USERNAME")
PROFILE = os.getenv("WSI_PROFILE_EMAIL")
PASSWORD = os.getenv("WSI_PASSWORD")

# API Base URL
WSI_BASE_SERVICE_URL = "https://www.wsitrader.com/Services/CSVDownloadService.svc"

# --- Define File Paths ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
base_project_dir = os.path.dirname(current_script_dir) # TRADERHELPER/
weather_csv_path = os.path.join(base_project_dir, "INFO", "WEATHER.csv")
# Corrected output path to be in the INFO folder
forecast_output_csv_path = os.path.join(base_project_dir, "INFO", "WEATHERforecast.csv")

# Expected columns for the final output
EXPECTED_FORECAST_COLUMNS = [
    'City Symbol', 'City Title', 'Date', 'Min Temp', 'Max Temp', 'Avg Temp',
    'CDD', 'HDD', '10yr Min Temp', '10yr Max Temp', '10yr Avg Temp',
    '10yr CDD', '10yr HDD'
]
# Columns to average for 10yr normals from historical data
TEN_YEAR_NORMAL_SOURCE_COLS = ['Min Temp', 'Max Temp', 'Avg Temp', 'CDD', 'HDD']
TEN_YEAR_NORMAL_TARGET_COLS = ['10yr Min Temp', '10yr Max Temp', '10yr Avg Temp', '10yr CDD', '10yr HDD']


def load_user_cities_from_weather_csv(csv_path):
    """
    Loads unique city symbols and their titles from the WEATHER.csv file.
    """
    print(f"Loading user cities from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"❌ ERROR: WEATHER.csv not found at {csv_path}")
        return {}, set()
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = df.columns.str.strip()
        if 'City Symbol' not in df.columns or 'City Title' not in df.columns:
            print("❌ ERROR: 'City Symbol' or 'City Title' not found in WEATHER.csv")
            return {}, set()
        
        city_title_map = df.drop_duplicates(subset=['City Symbol']) \
                           .set_index('City Symbol')['City Title'].to_dict()
        user_city_symbols = set(df['City Symbol'].unique())
        print(f"  Found {len(user_city_symbols)} unique city symbols in WEATHER.csv.")
        return city_title_map, user_city_symbols
    except Exception as e:
        print(f"❌ ERROR loading or processing WEATHER.csv: {e}")
        return {}, set()

def extract_city_info_from_full_name(name_series):
    """Helper to extract City Title and Symbol from 'City Name (SYMBOL)' format."""
    def extract(name):
        if pd.isna(name): return pd.NA, pd.NA
        name_str = str(name)
        match = re.search(r'\(([^)]+)\)$', name_str) 
        symbol = match.group(1) if match and match.group(1) else pd.NA 
        title = name_str.replace(f"({symbol})", "").strip() if pd.notna(symbol) else name_str
        return title, symbol
    return name_series.apply(lambda x: pd.Series(extract(x)))


def parse_min_max_forecast_data(raw_csv_text):
    """
    Parses the 'MinMax' CSV forecast from WSI GetCityTableForecast.
    Returns long format DataFrame with columns:
    ['City Symbol', 'City Title', 'Date', 'Min Temp', 'Max Temp']
    """
    try:
        lines = raw_csv_text.splitlines()
        if len(lines) < 4: 
            print("  ⚠️ Not enough lines in raw CSV to parse MinMax data (expected at least 4).")
            return pd.DataFrame()

        data_io = StringIO("\n".join(lines[1:]))
        df = pd.read_csv(data_io, header=[0, 1])
        df.columns = df.columns.map(lambda x: tuple(str(s).strip() for s in x))
        
        if len(df.columns) > 0:
            new_column_names_list = df.columns.tolist()
            new_column_names_list[0] = "CityFullName"
            df.columns = new_column_names_list
        else:
            print("  ⚠️ DataFrame has no columns after reading CSV with multi-header for MinMax.")
            return pd.DataFrame()

        value_vars_for_melt = [col for col in df.columns if col != "CityFullName"]
        df_melted = pd.melt(
            df, id_vars=["CityFullName"], value_vars=value_vars_for_melt,
            var_name="Date_Type_Tuple", value_name="Temperature"
        )

        df_melted['Date_Str_Header'] = df_melted['Date_Type_Tuple'].apply(lambda t: t[0] if isinstance(t, tuple) and len(t)>0 else None)
        df_melted['Temp_Type_Header'] = df_melted['Date_Type_Tuple'].apply(lambda t: t[1] if isinstance(t, tuple) and len(t)>1 else None)
        df_melted.drop(columns=['Date_Type_Tuple'], inplace=True)
        
        df_melted = df_melted[~df_melted['Date_Str_Header'].astype(str).str.contains('Normals', case=False, na=False)]
        df_melted = df_melted[~df_melted['Temp_Type_Header'].astype(str).str.contains('Normals', case=False, na=False)]
        df_melted = df_melted[~df_melted['Temp_Type_Header'].astype(str).str.fullmatch(r'\(F\)')]
        df_melted = df_melted[df_melted['Temp_Type_Header'].fillna('') != '']

        df_melted['Temperature'] = pd.to_numeric(df_melted['Temperature'], errors='coerce')
        df_melted.dropna(subset=['Temperature'], inplace=True)

        df_melted['Temp_Type'] = df_melted['Temp_Type_Header'].str.replace(':', '').str.strip()
        df_melted = df_melted[df_melted['Temp_Type'].isin(['Min', 'Max'])]

        df_melted['Date'] = pd.to_datetime(df_melted['Date_Str_Header'], format='%m/%d/%Y', errors='coerce')
        df_melted.dropna(subset=['Date'], inplace=True)

        df_pivot = df_melted.pivot_table(index=['CityFullName', 'Date'],
                                         columns='Temp_Type',
                                         values='Temperature').reset_index()
        
        if 'Min' not in df_pivot.columns: df_pivot['Min'] = np.nan
        if 'Max' not in df_pivot.columns: df_pivot['Max'] = np.nan
        df_pivot.rename(columns={'Min': 'Min Temp', 'Max': 'Max Temp'}, inplace=True)

        df_pivot[['City Title', 'City Symbol']] = extract_city_info_from_full_name(df_pivot['CityFullName'])
        
        final_cols = ['City Symbol', 'City Title', 'Date', 'Min Temp', 'Max Temp']
        for col in final_cols:
            if col not in df_pivot.columns: df_pivot[col] = np.nan
                
        result_df = df_pivot[final_cols].copy()
        result_df = result_df.dropna(subset=['City Symbol', 'Date']) 

        print(f"  Successfully parsed and transformed 'MinMax' data. Shape: {result_df.shape}")
        return result_df
    except Exception as e:
        print(f"  ❌ ERROR parsing MinMax data: {e}")
        import traceback; traceback.print_exc()
        return pd.DataFrame()

def parse_average_temp_forecast_data(raw_csv_text):
    """
    Parses the 'AverageTemp' CSV forecast from WSI GetCityTableForecast.
    Returns long format DataFrame with columns:
    ['City Symbol', 'City Title', 'Date', 'Avg Temp']
    """
    try:
        lines = raw_csv_text.splitlines()
        if len(lines) < 3: 
            print("  ⚠️ Not enough lines in raw CSV to parse AverageTemp data.")
            return pd.DataFrame()

        data_io = StringIO("\n".join(lines[1:])) 
        df = pd.read_csv(data_io, header=0) 
        df.columns = df.columns.str.strip()

        city_col_name = df.columns[0]
        df.rename(columns={city_col_name: "CityFullName"}, inplace=True)

        id_vars = ["CityFullName"]
        value_vars = [col for col in df.columns if col not in id_vars and not 'Normals' in str(col)]
        
        df_melted = pd.melt(df, id_vars=id_vars, value_vars=value_vars,
                            var_name='Date_Str_Header', value_name='Avg Temp')

        df_melted = df_melted[~df_melted['Date_Str_Header'].astype(str).str.contains('Normals', case=False, na=False)]
        
        df_melted['Avg Temp'] = pd.to_numeric(df_melted['Avg Temp'], errors='coerce')
        df_melted.dropna(subset=['Avg Temp'], inplace=True)

        df_melted['Date'] = pd.to_datetime(df_melted['Date_Str_Header'], format='%m/%d/%Y', errors='coerce')
        df_melted.dropna(subset=['Date'], inplace=True)

        df_melted[['City Title', 'City Symbol']] = extract_city_info_from_full_name(df_melted['CityFullName'])
        
        final_cols = ['City Symbol', 'City Title', 'Date', 'Avg Temp']
        for col in final_cols:
            if col not in df_melted.columns: df_melted[col] = np.nan
        
        result_df = df_melted[final_cols].copy()
        result_df = result_df.dropna(subset=['City Symbol', 'Date'])
        
        print(f"  Successfully parsed and transformed 'AverageTemp' data. Shape: {result_df.shape}")
        return result_df
    except Exception as e:
        print(f"  ❌ ERROR parsing AverageTemp data: {e}")
        import traceback; traceback.print_exc()
        return pd.DataFrame()

def parse_degree_days_forecast_data(raw_csv_text):
    """
    Parses the 'DegreeDays' CSV forecast from WSI GetCityTableForecast.
    Similar structure to MinMax, with HDD and CDD as sub-headers.
    Returns long format DataFrame with columns:
    ['City Symbol', 'City Title', 'Date', 'HDD', 'CDD']
    """
    try:
        lines = raw_csv_text.splitlines()
        if len(lines) < 4: 
            print("  ⚠️ Not enough lines in raw CSV to parse DegreeDays data.")
            return pd.DataFrame()

        # Preprocess the first header line (original line 1 of CSV)
        # If it ends with "Normals,", remove the trailing comma to balance with the second header line.
        # This addresses the "Header rows must have an equal number of columns" error.
        header_line_index_0 = 1 # This is lines[1]
        if lines[header_line_index_0].endswith("Normals,"):
            print(f"  Preprocessing DegreeDays header: Removing trailing comma from '{lines[header_line_index_0]}'")
            lines[header_line_index_0] = lines[header_line_index_0][:-1]
            print(f"  Processed DegreeDays header: '{lines[header_line_index_0]}'")


        data_io = StringIO("\n".join(lines[1:])) # Use preprocessed lines
        df = pd.read_csv(data_io, header=[0, 1]) 
        df.columns = df.columns.map(lambda x: tuple(str(s).strip() for s in x))
        
        if len(df.columns) > 0:
            new_column_names_list = df.columns.tolist()
            new_column_names_list[0] = "CityFullName"
            df.columns = new_column_names_list
        else:
            print("  ⚠️ DataFrame has no columns after reading CSV with multi-header for DegreeDays.")
            return pd.DataFrame()

        value_vars_for_melt = [col for col in df.columns if col != "CityFullName"]
        df_melted = pd.melt(
            df, id_vars=["CityFullName"], value_vars=value_vars_for_melt,
            var_name="Date_Type_Tuple", value_name="Value"
        )

        df_melted['Date_Str_Header'] = df_melted['Date_Type_Tuple'].apply(lambda t: t[0] if isinstance(t, tuple) and len(t)>0 else None)
        df_melted['DD_Type_Header'] = df_melted['Date_Type_Tuple'].apply(lambda t: t[1] if isinstance(t, tuple) and len(t)>1 else None)
        df_melted.drop(columns=['Date_Type_Tuple'], inplace=True)
        
        df_melted = df_melted[~df_melted['Date_Str_Header'].astype(str).str.contains('Normals', case=False, na=False)]
        df_melted = df_melted[~df_melted['DD_Type_Header'].astype(str).str.contains('Normals', case=False, na=False)]
        # For DegreeDays, sub-headers are HDD: CDD:, not (F)
        df_melted = df_melted[df_melted['DD_Type_Header'].fillna('') != ''] 


        df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce')
        df_melted.dropna(subset=['Value'], inplace=True)

        df_melted['DD_Type'] = df_melted['DD_Type_Header'].str.replace(':', '').str.strip().str.upper() 
        df_melted = df_melted[df_melted['DD_Type'].isin(['HDD', 'CDD'])]

        df_melted['Date'] = pd.to_datetime(df_melted['Date_Str_Header'], format='%m/%d/%Y', errors='coerce')
        df_melted.dropna(subset=['Date'], inplace=True)

        df_pivot = df_melted.pivot_table(index=['CityFullName', 'Date'],
                                         columns='DD_Type',
                                         values='Value').reset_index()
        
        if 'HDD' not in df_pivot.columns: df_pivot['HDD'] = np.nan
        if 'CDD' not in df_pivot.columns: df_pivot['CDD'] = np.nan
        
        df_pivot[['City Title', 'City Symbol']] = extract_city_info_from_full_name(df_pivot['CityFullName'])
        
        final_cols = ['City Symbol', 'City Title', 'Date', 'HDD', 'CDD']
        for col in final_cols:
            if col not in df_pivot.columns: df_pivot[col] = np.nan
                
        result_df = df_pivot[final_cols].copy()
        result_df = result_df.dropna(subset=['City Symbol', 'Date'])

        print(f"  Successfully parsed and transformed 'DegreeDays' data. Shape: {result_df.shape}")
        return result_df
    except pd.errors.ParserError as pe: 
        print(f"  ❌ pandas.errors.ParserError parsing DegreeDays data: {pe}")
        print("     This often means the header structure (e.g., number of columns in header rows) is unexpected.")
        print("     Inspect the raw CSV for 'DegreeDays' to confirm its header layout.")
        import traceback; traceback.print_exc()
        return pd.DataFrame()
    except Exception as e:
        print(f"  ❌ ERROR parsing DegreeDays data: {e}")
        import traceback; traceback.print_exc()
        return pd.DataFrame()


def call_get_city_table_forecast_api(tab_name, region="NA", site_id="allcities"):
    """
    Calls the GetCityTableForecast API for a specific CurrentTabName.
    """
    print(f"\nFetching '{tab_name}' forecast data from WSI API (GetCityTableForecast)...")
    api_url = f"{WSI_BASE_SERVICE_URL}/GetCityTableForecast"
    params = {
        "Account": USERNAME,
        "Profile": PROFILE,
        "Password": PASSWORD,
        "SiteId": site_id,
        "IsCustom": "false", 
        "CurrentTabName": tab_name,
        "TempUnits": "F",
        "Region": region
    }

    if not all([USERNAME, PROFILE, PASSWORD]):
        print("❌ ERROR: WSI credentials not loaded. Check .env file.")
        return None 

    try:
        response = requests.get(api_url, params=params, timeout=180) 
        response.raise_for_status()
        
        raw_csv_text = response.text
        # Print raw response for all calls now for better debugging
        print(f"--- Raw API Response for '{tab_name}' (first 1000 characters) ---")
        print(raw_csv_text[:1000])
        print("------------------------------------------------------------------\n")

        if not raw_csv_text.strip() or raw_csv_text.count('\n') < 3: 
            print(f"  ⚠️ API response for '{tab_name}' has too few lines or is empty.")
            return None
        if "no data" in raw_csv_text.lower() or "Error" in raw_csv_text.lower():
            print(f"  ⚠️ 'no data' or 'Error' string found in API response for '{tab_name}'.")
            return None
        
        parsed_df = None
        if tab_name == "MinMax":
            parsed_df = parse_min_max_forecast_data(raw_csv_text)
        elif tab_name == "AverageTemp":
            parsed_df = parse_average_temp_forecast_data(raw_csv_text)
        elif tab_name == "DegreeDays":
            parsed_df = parse_degree_days_forecast_data(raw_csv_text)
        else:
            print(f"  ⚠️ Parsing for tab_name '{tab_name}' is not implemented.")
            return None

        if parsed_df is None or parsed_df.empty: 
             print(f"  ⚠️ DataFrame is empty or None after parsing API response for '{tab_name}'.")
             return None
        
        print(f"--- Parsed & Transformed DataFrame Info for '{tab_name}' ---")
        parsed_df.info(verbose=False, show_counts=True) 
        print(f"\n--- Parsed & Transformed DataFrame Head for '{tab_name}' (first 3 rows) ---")
        print(parsed_df.head(3).to_string())
        print("------------------------------------------------------------------\n")
        return parsed_df

    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR during API request for '{tab_name}': {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}, Text: {e.response.text[:200]}")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred for '{tab_name}': {e}")
        import traceback; traceback.print_exc()
        return None

def calculate_10yr_normals_from_historical(historical_df, forecast_dates, cities_map):
    """
    Calculates 10-year normals using the provided historical DataFrame.
    """
    print("\nCalculating 10-year normals from historical data...")
    if historical_df.empty:
        print("  Historical data is empty. Cannot calculate 10-year normals.")
        return pd.DataFrame(columns=['City Symbol', 'City Title', 'Date'] + TEN_YEAR_NORMAL_TARGET_COLS)

    hist_df = historical_df.copy()
    hist_df['Date'] = pd.to_datetime(hist_df['Date'])
    hist_df['MonthDay'] = hist_df['Date'].dt.strftime('%m-%d')
    hist_df['Year'] = hist_df['Date'].dt.year

    all_normals = []

    for city_symbol, city_title in cities_map.items():
        for forecast_date in forecast_dates:
            month_day_str = forecast_date.strftime('%m-%d')
            current_year = forecast_date.year
            
            city_historical_data = hist_df[
                (hist_df['City Symbol'] == city_symbol) &
                (hist_df['MonthDay'] == month_day_str) &
                (hist_df['Year'] >= current_year - 10) &
                (hist_df['Year'] < current_year) 
            ]
            
            normal_values = {'City Symbol': city_symbol, 'City Title': city_title, 'Date': forecast_date}
            if not city_historical_data.empty:
                for i, source_col in enumerate(TEN_YEAR_NORMAL_SOURCE_COLS):
                    target_col = TEN_YEAR_NORMAL_TARGET_COLS[i]
                    if source_col in city_historical_data.columns:
                        mean_val = pd.to_numeric(city_historical_data[source_col], errors='coerce').mean(skipna=True)
                        normal_values[target_col] = mean_val if pd.notna(mean_val) else np.nan
                    else:
                        normal_values[target_col] = np.nan
            else: 
                for target_col in TEN_YEAR_NORMAL_TARGET_COLS:
                    normal_values[target_col] = np.nan
            all_normals.append(normal_values)
    
    if not all_normals:
        print("  No normals calculated (perhaps no matching historical data or forecast dates).")
        return pd.DataFrame(columns=['City Symbol', 'City Title', 'Date'] + TEN_YEAR_NORMAL_TARGET_COLS)

    normals_df = pd.DataFrame(all_normals)
    print(f"  Successfully calculated 10-year normals. Shape: {normals_df.shape}")
    return normals_df


def main():
    print("--- Starting Weather Forecast Generation (Using GetCityTableForecast) ---")
    
    user_city_titles_map, user_cities_to_forecast = load_user_cities_from_weather_csv(weather_csv_path)
    if not user_cities_to_forecast:
        print("No user cities loaded from WEATHER.csv. Exiting.")
        return

    min_max_df = call_get_city_table_forecast_api(tab_name="MinMax")
    avg_temp_df = call_get_city_table_forecast_api(tab_name="AverageTemp")
    degree_days_df = call_get_city_table_forecast_api(tab_name="DegreeDays")
    
    if min_max_df is None or min_max_df.empty:
        print("MinMax forecast data is missing or empty. Cannot proceed with primary data merge.")
        return
    
    forecast_df = min_max_df.copy()

    if avg_temp_df is not None and not avg_temp_df.empty:
        forecast_df = pd.merge(forecast_df, avg_temp_df, on=['City Symbol', 'City Title', 'Date'], how='outer')
    else:
        print("  ⚠️ AverageTemp data missing or empty, 'Avg Temp' will be NaN in the merged DataFrame.")
        if 'Avg Temp' not in forecast_df.columns: forecast_df['Avg Temp'] = np.nan

    if degree_days_df is not None and not degree_days_df.empty:
        forecast_df = pd.merge(forecast_df, degree_days_df, on=['City Symbol', 'City Title', 'Date'], how='outer')
    else:
        print("  ⚠️ DegreeDays data missing or empty, 'HDD'/'CDD' will be NaN in the merged DataFrame.")
        if 'HDD' not in forecast_df.columns: forecast_df['HDD'] = np.nan
        if 'CDD' not in forecast_df.columns: forecast_df['CDD'] = np.nan

    if forecast_df.empty: 
        print("No forecast data available after attempting to fetch all types. Exiting.")
        return
        
    forecast_df_filtered = forecast_df[forecast_df['City Symbol'].isin(user_cities_to_forecast)].copy()
    
    if forecast_df_filtered.empty:
        print("No forecast data available for the cities specified in your WEATHER.csv after filtering. Exiting.")
        return
    
    print(f"\n--- Combined and Filtered Forecast Data (for user cities) ---")
    forecast_df_filtered.info(verbose=False, show_counts=True)
    print(forecast_df_filtered.head(3).to_string())

    historical_data_df = pd.read_csv(weather_csv_path, low_memory=False)
    if historical_data_df.empty:
        print("  ⚠️ Historical data (WEATHER.csv) is empty. 10-year normals will be NaN.")
        normals_df = pd.DataFrame() 
    else:
        historical_data_df.columns = historical_data_df.columns.str.strip()
        today = datetime.now().date()
        forecast_end_date = today + timedelta(days=14) 
        
        forecast_df_filtered['Date'] = pd.to_datetime(forecast_df_filtered['Date']).dt.date
        forecast_df_for_normals = forecast_df_filtered[
            (forecast_df_filtered['Date'] >= today) &
            (forecast_df_filtered['Date'] <= forecast_end_date)
        ].copy() 
        
        if forecast_df_for_normals.empty:
            print("No forecast data within the desired 15-day range starting today for normal calculations.")
            normals_df = pd.DataFrame()
        else:
            unique_forecast_dates = pd.to_datetime(forecast_df_for_normals['Date'].unique()).normalize()
            cities_in_filtered_forecast_map = {
                sym: forecast_df_for_normals[forecast_df_for_normals['City Symbol'] == sym]['City Title'].iloc[0]
                for sym in forecast_df_for_normals['City Symbol'].unique()
            }
            normals_df = calculate_10yr_normals_from_historical(historical_data_df, unique_forecast_dates, cities_in_filtered_forecast_map)

    if not normals_df.empty:
        forecast_df_filtered['Date'] = pd.to_datetime(forecast_df_filtered['Date']) 
        normals_df['Date'] = pd.to_datetime(normals_df['Date']) 
        final_df = pd.merge(forecast_df_filtered, normals_df, on=['City Symbol', 'City Title', 'Date'], how='left')
    else:
        print("  ⚠️ No 10-year normals were calculated or merged. Normals columns will be NaN.")
        final_df = forecast_df_filtered.copy()
        for col in TEN_YEAR_NORMAL_TARGET_COLS:
            if col not in final_df.columns:
                final_df[col] = np.nan
    
    for col in EXPECTED_FORECAST_COLUMNS:
        if col not in final_df.columns:
            final_df[col] = np.nan 
    
    final_df = final_df[EXPECTED_FORECAST_COLUMNS] 
    final_df.sort_values(by=['City Symbol', 'Date'], inplace=True)
    
    try:
        final_df['Date'] = pd.to_datetime(final_df['Date']).dt.strftime('%Y-%m-%d')
        final_df.to_csv(forecast_output_csv_path, index=False)
        print(f"\n✅ Successfully saved forecast data to: {forecast_output_csv_path}")
        print(f"   Forecast covers {final_df['Date'].nunique()} days for {final_df['City Symbol'].nunique()} cities.")
        print(f"   Total rows: {len(final_df)}")
        print("\n--- Final Forecast Data Head (first 5 rows) ---")
        print(final_df.head().to_string())

    except Exception as e:
        print(f"\n❌ Error saving forecast data to CSV: {e}")

if __name__ == "__main__":
    main()