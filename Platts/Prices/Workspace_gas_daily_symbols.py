import requests
import pandas as pd
import os
import time
from datetime import datetime, timedelta

# --- Configuration ---
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file
USERNAME = os.getenv("PLATTS_USERNAME")
PASSWORD = os.getenv("PLATTS_PASSWORD")

# API Endpoints
AUTH_URL = "https://api.ci.spglobal.com/auth/api"
REFERENCE_DATA_SEARCH_URL = "https://api.ci.spglobal.com/market-data/reference-data/v3/search"
MARKET_DATA_HISTORY_URL = "https://api.ci.spglobal.com/market-data/v3/value/history/symbol"

# Symbol Fetching Configuration
SYMBOL_API_Q_KEYWORD = "Natural Gas"
FIELDS_TO_RETRIEVE_SYMBOLS = ["symbol", "description", "commodity", "uom", "currency"]
LOCAL_DESCRIPTION_FILTER_ENDSWITH = "FDt Com" 
PRELIM_FILTER_EXCLUDE_KEYWORD = "Prelim" # Keyword to exclude from descriptions

# Historical Data Fetching Configuration
BATES_TO_RETRIEVE_HISTORY_ORIGINAL = "c,u,l,h,o" # Close, Index, Low, High, Open. 'v' removed due to previous errors.
BATES_FOR_HENRY_HUB_TEST = "u" # Kept for reference for the test function

# Output files
RAW_SYMBOLS_OUTPUT_CSV_FILE = "all_natural_gas_symbols_temp.csv"
FILTERED_SYMBOLS_OUTPUT_CSV_FILE = "filtered_symbols_for_history.csv"
HISTORICAL_DATA_OUTPUT_CSV_FILE = "historical_market_data.csv" 

PAGE_SIZE = 100 
HISTORY_PAGE_SIZE = 1000 
SUBSCRIBED_ONLY = True

def get_access_token(username, password):
    """Authenticates with the S&P Global Platts API and retrieves an access token."""
    if not username or not password:
        print("ERROR: Username or Password not found. Please ensure .env file is set up correctly with PLATTS_USERNAME and PLATTS_PASSWORD.")
        return None
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    payload = {"username": username, "password": password}
    print("Attempting to obtain access token...")
    try:
        response = requests.post(AUTH_URL, headers=headers, data=payload, timeout=30)
        response.raise_for_status() 
        token_data = response.json()
        print("Successfully obtained access token.")
        return token_data.get("access_token")
    except requests.exceptions.HTTPError as http_err:
        print(f"Http Error during authentication: {http_err}")
        if http_err.response is not None:
            print(f"Response status: {http_err.response.status_code}")
            print(f"Response content: {http_err.response.content.decode()}")
    except Exception as e:
        print(f"An unexpected error occurred during authentication: {e}")
    return None

def search_symbols_api(token, q_keyword, fields, page_size=100, subscribed_only=True):
    """Searches for symbols using the Reference Data API with the 'q' keyword parameter and handles pagination."""
    if not token: 
        print("No access token provided for symbol search.")
        return []
        
    all_symbols_data = []
    page = 1
    total_pages = 1 
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    print(f"\nStarting API symbol search with keyword (q): '{q_keyword}'")
    
    while page <= total_pages:
        params = {
            "q": q_keyword, 
            "Field": ",".join(fields), 
            "PageSize": page_size, 
            "Page": page, 
            "subscribed_only": str(subscribed_only).lower()
        }
        try:
            print(f"Fetching page {page} of symbols from API...")
            response = requests.get(REFERENCE_DATA_SEARCH_URL, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if results:
                all_symbols_data.extend(results)
                print(f"Fetched {len(results)} symbols from API on page {page}.")
            
            metadata = data.get("metadata", {})
            current_total_pages = metadata.get('totalPages', metadata.get('total_pages'))
            if current_total_pages is not None:
                total_pages = current_total_pages
            elif page == 1 and not results: 
                total_pages = 0 
            
            if page >= total_pages or not results: 
                break
            page += 1
            time.sleep(0.1) 

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP Error during API symbol search (page {page}): {http_err}")
            if http_err.response is not None:
                print(f"Response status: {http_err.response.status_code}")
                print(f"Response content: {http_err.response.content.decode()}")
            break 
        except Exception as e:
            print(f"An unexpected error during API symbol search (page {page}): {e}")
            break
    print(f"Total symbols data fetched from API: {len(all_symbols_data)}")
    return all_symbols_data

def get_historical_data(token, symbol, assess_date_from, assess_date_to, bates_str, fields_to_request_str=None):
    """Fetches historical market data for a given symbol, date range, and bates."""
    if not token: 
        print(f"No access token provided for historical data fetch for symbol {symbol}.")
        return []
    
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    
    bates_for_filter_parts = [f'bate:"{b.strip()}"' for b in bates_str.split(',')]
    bates_segment = " OR ".join(bates_for_filter_parts)
    
    custom_filter = (
        f'symbol:"{symbol}" AND ({bates_segment}) ' 
        f'AND assessDate:["{assess_date_from}" TO "{assess_date_to}"]'
    )
    
    params = {
        "Filter": custom_filter,
        "PageSize": HISTORY_PAGE_SIZE
    }
    if fields_to_request_str and fields_to_request_str.strip(): 
        params["Field"] = fields_to_request_str
    
    all_history_for_symbol = []
    page = 1
    total_pages = 1
    
    print(f"Fetching historical data for symbol: {symbol}, From: {assess_date_from}, To: {assess_date_to}, Bates: {bates_str}")
    if "Field" in params: 
        print(f"  Requesting Fields: {params['Field']}")
    else:
        print("  Requesting default fields from API (Field parameter omitted).")
    print(f"  Using Custom Filter: {custom_filter}")
    
    while page <= total_pages:
        params["Page"] = page
        try:
            response = requests.get(MARKET_DATA_HISTORY_URL, headers=headers, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if results:
                all_history_for_symbol.extend(results)
                print(f"  Fetched {len(results)} records for {symbol} on page {page}.")
            
            metadata = data.get("metadata", {})
            current_total_pages = metadata.get('totalPages', metadata.get('total_pages'))
            if current_total_pages is not None:
                total_pages = current_total_pages
            elif page == 1 and not results:
                total_pages = 0
            
            if page >= total_pages or not results:
                break
            page += 1
            time.sleep(0.1)

        except requests.exceptions.HTTPError as http_err:
            print(f"  HTTP Error fetching history for symbol {symbol} (page {page}): {http_err}")
            if http_err.response is not None:
                print(f"  Response status: {http_err.response.status_code}")
                print(f"  Response content: {http_err.response.content.decode()}")
            break 
        except Exception as e:
            print(f"  General Error fetching history for symbol {symbol} (page {page}): {e}")
            break
            
    return all_history_for_symbol

def main_test_henry_hub(): # Kept for reference, but original_main will be called
    """Test Henry Hub historical data retrieval using a valid symbol and bates."""
    if not USERNAME or not PASSWORD:
        print("ERROR: Credentials not loaded from .env file. Ensure PLATTS_USERNAME and PLATTS_PASSWORD are set.")
        return

    access_token = get_access_token(USERNAME, PASSWORD)
    if not access_token:
        print("Could not proceed without an access token.")
        return

    test_symbol = "IGBDU21" 
    test_description = "Henry Hub FDt Com" 
    
    bates_to_use = BATES_FOR_HENRY_HUB_TEST 

    today = datetime(2025, 5, 25) 
    end_date_obj = today - timedelta(days=1)
    start_date_obj = end_date_obj - timedelta(days=44) 

    test_start_date = start_date_obj.strftime("%Y-%m-%d")
    test_end_date = end_date_obj.strftime("%Y-%m-%d")


    print(f"\n--- Starting Test for Henry Hub (Symbol: {test_symbol}) ---")
    print(f"--- Dynamic Date Range: {test_start_date} to {test_end_date} (Last 45 days) ---")
    print(f"--- Requesting Bates: {bates_to_use} ---")


    all_historical_data_list = []
    history = get_historical_data(access_token, test_symbol, test_start_date, test_end_date, bates_to_use) 
    if history:
        for record in history:
            record['symbol'] = test_symbol
            record['description'] = test_description
        all_historical_data_list.extend(history)

    if not all_historical_data_list:
        print(f"\nNo historical data found for {test_symbol} in the range {test_start_date} to {test_end_date}.")
        return

    df_historical_data = pd.DataFrame(all_historical_data_list)
    print(f"\nFetched a total of {len(df_historical_data)} historical records for {test_symbol}.")
    
    if not df_historical_data.empty:
        print("\nSample of raw fetched records (first 2 to inspect default fields):")
        if len(df_historical_data) > 0:
            print("First record detail:", df_historical_data.iloc[0].to_dict())
        else:
            print("No records to show sample.")


    processed_records = []
    for index, row_data in df_historical_data.iterrows():
        record = {
            'symbol': row_data.get('symbol'), 
            'description': row_data.get('description'), 
            'assessDate': row_data.get('assessDate'), 
            'currency': row_data.get('currency'), 
            'uom': row_data.get('uom') 
        }
        if 'bateValues' in row_data and isinstance(row_data['bateValues'], list):
            for bv in row_data['bateValues']:
                bate = bv.get('bate')
                value = bv.get('value')
                if bate:
                    record[f'bate_{bate}'] = value
        processed_records.append(record)

    if not processed_records:
        print("No data after processing bateValues. Check API response structure.")
        return

    df_final_data = pd.DataFrame(processed_records)

    def determine_primary_price(row_data_func):
        description_val = row_data_func.get('description', '')
        if isinstance(description_val, str) and description_val.lower().startswith('ice'):
            return row_data_func.get('bate_c') 
        else:
            return row_data_func.get('bate_u')

    if 'description' in df_final_data.columns:
        df_final_data['PrimaryPrice'] = df_final_data.apply(determine_primary_price, axis=1)

    column_renames = {
        'assessDate': 'Date',
        'bate_u': 'Index(u)', 
    }
    df_final_data.rename(columns=column_renames, inplace=True)

    desired_columns_order = [
        'symbol', 'description', 'Date', 'currency', 'uom',
        'Index(u)', 'PrimaryPrice' 
    ]

    final_columns_present = [col for col in desired_columns_order if col in df_final_data.columns]
    df_final_data_subset = df_final_data[final_columns_present]

    try:
        df_final_data_subset.to_csv(HISTORICAL_DATA_OUTPUT_CSV_FILE, index=False)
        print(f"\nSuccessfully saved historical data for {test_symbol} to {HISTORICAL_DATA_OUTPUT_CSV_FILE}")
        print("\nSample of the data:")
        print(df_final_data_subset.head())
    except Exception as e:
        print(f"Error saving final historical data to CSV {HISTORICAL_DATA_OUTPUT_CSV_FILE}: {e}")


def original_main(): 
    """Original main function to orchestrate symbol fetching, user selection, and historical data retrieval."""
    if not USERNAME or not PASSWORD:
        print("ERROR: Credentials not loaded from .env file. Ensure PLATTS_USERNAME and PLATTS_PASSWORD are set.")
        return

    access_token = get_access_token(USERNAME, PASSWORD)
    if not access_token:
        print("Could not proceed without an access token.")
        return

    # --- Part 1: Fetch and Filter Symbols ---
    raw_symbols_data = search_symbols_api(
        token=access_token,
        q_keyword=SYMBOL_API_Q_KEYWORD,
        fields=FIELDS_TO_RETRIEVE_SYMBOLS,
        page_size=PAGE_SIZE,
        subscribed_only=SUBSCRIBED_ONLY
    )

    if not raw_symbols_data:
        print(f"No symbols data was fetched from the API using keyword '{SYMBOL_API_Q_KEYWORD}'. Exiting.")
        return
        
    df_raw_symbols = pd.DataFrame(raw_symbols_data)
    print(f"\nFetched {len(df_raw_symbols)} total symbols using API keyword '{SYMBOL_API_Q_KEYWORD}'.")
    
    if 'description' not in df_raw_symbols.columns:
        print("Error: 'description' column not found in fetched symbols. Cannot proceed with filtering.")
        return

    df_raw_symbols['description'] = df_raw_symbols['description'].astype(str)
    
    df_symbols_fdt_com = df_raw_symbols[
        df_raw_symbols['description'].str.lower().str.endswith(LOCAL_DESCRIPTION_FILTER_ENDSWITH.lower(), na=False)
    ]
    print(f"Filtered down to {len(df_symbols_fdt_com)} symbols with description ending with (case-insensitive) '{LOCAL_DESCRIPTION_FILTER_ENDSWITH}'.")

    if df_symbols_fdt_com.empty:
        print(f"No symbols matched the '{LOCAL_DESCRIPTION_FILTER_ENDSWITH}' description filter. Exiting.")
        return
        
    df_symbols_final_filtered = df_symbols_fdt_com[
        ~df_symbols_fdt_com['description'].str.lower().str.contains(PRELIM_FILTER_EXCLUDE_KEYWORD.lower(), na=False)
    ]
    print(f"Further filtered to {len(df_symbols_final_filtered)} symbols by excluding descriptions containing (case-insensitive) '{PRELIM_FILTER_EXCLUDE_KEYWORD}'.")

    if df_symbols_final_filtered.empty:
        print("No symbols remained after excluding 'Prelim' descriptions. Exiting.")
        return
        
    try:
        df_symbols_final_filtered.to_csv(FILTERED_SYMBOLS_OUTPUT_CSV_FILE, index=False)
        print(f"Saved {len(df_symbols_final_filtered)} final filtered symbols to {FILTERED_SYMBOLS_OUTPUT_CSV_FILE}")
    except Exception as e:
        print(f"Error saving filtered symbols to CSV {FILTERED_SYMBOLS_OUTPUT_CSV_FILE}: {e}")


    unique_descriptions = sorted(df_symbols_final_filtered['description'].unique())
    if not unique_descriptions: 
        print("No unique descriptions found from final filtered symbols. Exiting.")
        return

    print("\nAvailable final filtered descriptions for historical data:")
    for i, desc in enumerate(unique_descriptions):
        print(f"{i+1}. {desc}")
    
    selected_indices_str = input(f"Enter the numbers of the descriptions you want to process (comma-separated, e.g., 1,3,5), or type 'all' to process all: ")
    
    if selected_indices_str.strip().lower() == 'all':
        selected_descriptions = unique_descriptions
        print("Processing all available symbols.")
    else:
        try:
            selected_indices = [int(x.strip()) - 1 for x in selected_indices_str.split(',')]
            selected_descriptions = [unique_descriptions[i] for i in selected_indices if 0 <= i < len(unique_descriptions)]
        except ValueError:
            print("Invalid input for selection. Please enter numbers separated by commas or 'all'.")
            return

    if not selected_descriptions:
        print("No valid descriptions selected. Exiting.")
        return

    print(f"\nYou selected the following descriptions for historical data retrieval: {', '.join(selected_descriptions)}")
    df_selected_symbols = df_symbols_final_filtered[df_symbols_final_filtered['description'].isin(selected_descriptions)]

    # For original_main, prompt for dates instead of calculating dynamically
    while True:
        start_date_str_orig = input("Enter start date for original main (YYYY-MM-DD): ")
        end_date_str_orig = input("Enter end date for original main (YYYY-MM-DD): ")
        try:
            start_date_obj_orig = datetime.strptime(start_date_str_orig, "%Y-%m-%d")
            end_date_obj_orig = datetime.strptime(end_date_str_orig, "%Y-%m-%d")
            if start_date_obj_orig > end_date_obj_orig:
                print("Start date cannot be after end date.")
            else:
                break
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")


    all_historical_data_list = []
    for index, row in df_selected_symbols.iterrows():
        symbol = row['symbol']
        description = row['description'] 
        
        time.sleep(0.3) 
        
        # Call get_historical_data without the Field parameter for the original main workflow as well
        # Using BATES_TO_RETRIEVE_HISTORY_ORIGINAL which is "c,u,l,h,o"
        history = get_historical_data(access_token, symbol, start_date_str_orig, end_date_str_orig, BATES_TO_RETRIEVE_HISTORY_ORIGINAL) 
        if history:
            for record in history:
                record['symbol'] = symbol
                record['description'] = description 
            all_historical_data_list.extend(history)

    if not all_historical_data_list:
        print("\nNo historical data found for the selected symbols and date range.")
        return

    df_historical_data = pd.DataFrame(all_historical_data_list)
    print(f"\nFetched a total of {len(df_historical_data)} historical records (may include multiple bates per day).")
    
    # Print sample of raw historical data to inspect default fields
    if not df_historical_data.empty:
        print("\nSample of raw fetched historical records (first 2 to inspect default fields):")
        # Temporarily print more details of the first record to see its structure
        if len(df_historical_data) > 0:
            print("First record detail:", df_historical_data.iloc[0].to_dict())
        else:
            print("No records to show sample.")

    processed_records = []
    for index, row_data in df_historical_data.iterrows():
        record = {
            'symbol': row_data.get('symbol'), 
            'description': row_data.get('description'), 
            'assessDate': row_data.get('assessDate'), 
            'currency': row_data.get('currency'), 
            'uom': row_data.get('uom')
        }
        if 'bateValues' in row_data and isinstance(row_data['bateValues'], list):
            for bv in row_data['bateValues']:
                bate = bv.get('bate')
                value = bv.get('value')
                if bate: 
                    record[f'bate_{bate}'] = value 
        processed_records.append(record)
    
    if not processed_records:
        print("No data after processing bateValues. Check API response structure for historical data.")
        return

    df_final_data = pd.DataFrame(processed_records)
    
    def determine_primary_price(row_data_func):
        description_val = row_data_func.get('description', '') 
        if isinstance(description_val, str) and description_val.lower().startswith('ice'):
            return row_data_func.get('bate_c') 
        else:
            return row_data_func.get('bate_u') 

    if 'description' in df_final_data.columns: 
       df_final_data['PrimaryPrice'] = df_final_data.apply(determine_primary_price, axis=1)

    column_renames = {
        'assessDate': 'Date',
        'bate_l': 'Low(l)',
        'bate_h': 'High(h)',
        'bate_c': 'Close(c)',
        'bate_u': 'Index(u)',
        'bate_o': 'Open(o)',
        # 'bate_v': 'Volume(v)' # 'v' is not in BATES_TO_RETRIEVE_HISTORY_ORIGINAL anymore
    }
    df_final_data.rename(columns=column_renames, inplace=True)

    desired_columns_order = [
        'symbol', 'description', 'Date', 'currency', 'uom',
        'Low(l)', 'High(h)', 'Open(o)', 'Close(c)', 'Index(u)', 'PrimaryPrice' # Removed Volume(v)
    ]
    
    final_columns_present = [col for col in desired_columns_order if col in df_final_data.columns]
    df_final_data_subset = df_final_data[final_columns_present]

    try:
        df_final_data_subset.to_csv(HISTORICAL_DATA_OUTPUT_CSV_FILE, index=False)
        print(f"\nSuccessfully saved historical data for selected symbols to {HISTORICAL_DATA_OUTPUT_CSV_FILE}")
    except Exception as e:
        print(f"Error saving final historical data to CSV {HISTORICAL_DATA_OUTPUT_CSV_FILE}: {e}")

    print("\nCharting functionality to be added in a future step.")


if __name__ == "__main__":
    # To run the Henry Hub test:
    # main_test_henry_hub() 

    # To run the original full workflow:
    original_main()
