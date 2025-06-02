import requests
import pandas as pd
import os
import time
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv() # Load environment variables from .env file
USERNAME = os.getenv("PLATTS_USERNAME")
PASSWORD = os.getenv("PLATTS_PASSWORD")

# API Endpoints
AUTH_URL = "https://api.ci.spglobal.com/auth/api"
REFERENCE_DATA_SEARCH_URL = "https://api.ci.spglobal.com/market-data/reference-data/v3/search"

# Symbol Fetching Configuration
SYMBOL_API_Q_KEYWORD = "Natural Gas" # Broad filter for the API call
FIELDS_TO_RETRIEVE = ["symbol", "description", "commodity", "uom", "currency"] # Added a few more for context, can be trimmed later
PAGE_SIZE = 1000 
SUBSCRIBED_ONLY = True 

# Local Filtering Criteria
LOCAL_DESCRIPTION_FILTER_ENDSWITH = "fdt com" # Case-insensitive
PRELIM_FILTER_EXCLUDE_KEYWORD = "prelim" # Case-insensitive

# Output Configuration
OUTPUT_FOLDER_NAME = "INFO"
OUTPUT_FILE_NAME = "PlattsSymbols.csv"
OUTPUT_CSV_FULL_PATH = os.path.join(OUTPUT_FOLDER_NAME, OUTPUT_FILE_NAME)


def get_access_token(username, password):
    """Authenticates and retrieves an access token."""
    if not username or not password:
        print("ERROR: Username or Password not found in .env file. Please ensure PLATTS_USERNAME and PLATTS_PASSWORD are set.")
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

def fetch_symbols_by_keyword(token, q_keyword, fields, page_size=1000, subscribed_only=True):
    """
    Fetches symbols based on a keyword search using the Reference Data API.
    """
    if not token: 
        print("No access token provided for symbol search.")
        return []
        
    all_symbols_data = []
    page = 1
    total_pages = 1 
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    print(f"\nStarting API symbol search with keyword (q): '{q_keyword}'.")
    print(f"Retrieving fields: {', '.join(fields)}")
    
    while page <= total_pages:
        params = {
            "q": q_keyword, # Using keyword search
            "Field": ",".join(fields), 
            "PageSize": page_size, 
            "Page": page, 
            "subscribed_only": str(subscribed_only).lower()
        }
        try:
            print(f"Fetching page {page} of symbols...")
            response = requests.get(REFERENCE_DATA_SEARCH_URL, headers=headers, params=params, timeout=60) 
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if results:
                all_symbols_data.extend(results)
                print(f"Fetched {len(results)} symbols on page {page}.")
            
            metadata = data.get("metadata", {})
            current_total_pages = metadata.get('totalPages', metadata.get('total_pages'))
            if current_total_pages is not None:
                total_pages = current_total_pages
            elif page == 1 and not results: 
                total_pages = 0 
                print(f"No symbols found for keyword '{q_keyword}' on the first page.")
            
            if page >= total_pages or not results: 
                break
            page += 1
            time.sleep(0.2) 

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP Error during API symbol search (page {page}): {http_err}")
            if hasattr(http_err, 'response') and http_err.response is not None:
                print(f"Response status: {http_err.response.status_code}")
                print(f"Response content: {http_err.response.content.decode()}")
            break 
        except Exception as e:
            print(f"An unexpected error during API symbol search (page {page}): {e}")
            break
    print(f"Total symbols data fetched from API for keyword '{q_keyword}': {len(all_symbols_data)}")
    return all_symbols_data

def main():
    """Main function to fetch specific symbols and save them."""
    if not USERNAME or not PASSWORD:
        print("ERROR: Credentials not loaded from .env file. Ensure PLATTS_USERNAME and PLATTS_PASSWORD are set.")
        return

    access_token = get_access_token(USERNAME, PASSWORD)
    if not access_token:
        print("Could not proceed without an access token.")
        return

    print(f"Attempting to fetch 'Natural Gas' symbols for further filtering.")
    raw_symbols_list = fetch_symbols_by_keyword( # Changed function name for clarity
        token=access_token,
        q_keyword=SYMBOL_API_Q_KEYWORD,
        fields=FIELDS_TO_RETRIEVE,
        page_size=PAGE_SIZE,
        subscribed_only=SUBSCRIBED_ONLY
    )

    if raw_symbols_list:
        df_raw_symbols = pd.DataFrame(raw_symbols_list)
        
        df_filtered_symbols = df_raw_symbols.copy() # Start with a copy

        # Apply local filters
        if 'description' in df_filtered_symbols.columns:
            df_filtered_symbols['description'] = df_filtered_symbols['description'].astype(str)
            
            # Filter 1: ends with "FDt Com" (case-insensitive)
            initial_count = len(df_filtered_symbols)
            df_filtered_symbols = df_filtered_symbols[
                df_filtered_symbols['description'].str.lower().str.endswith(LOCAL_DESCRIPTION_FILTER_ENDSWITH, na=False)
            ]
            print(f"Filtered from {initial_count} to {len(df_filtered_symbols)} symbols ending with '{LOCAL_DESCRIPTION_FILTER_ENDSWITH}'.")

            # Filter 2: does not contain "Prelim" (case-insensitive)
            if not df_filtered_symbols.empty:
                initial_count_after_fdt = len(df_filtered_symbols)
                df_filtered_symbols = df_filtered_symbols[
                    ~df_filtered_symbols['description'].str.lower().str.contains(PRELIM_FILTER_EXCLUDE_KEYWORD, na=False)
                ]
                print(f"Further filtered from {initial_count_after_fdt} to {len(df_filtered_symbols)} symbols by excluding '{PRELIM_FILTER_EXCLUDE_KEYWORD}'.")
        else:
            print("Warning: 'description' column not found in fetched symbols. Cannot apply description-based filters.")
        
        # Select only symbol and description for the final output as per original specific request
        df_final_output = df_filtered_symbols[['symbol', 'description']]


        if not df_final_output.empty:
            # Ensure the output folder exists
            if not os.path.exists(OUTPUT_FOLDER_NAME):
                try:
                    os.makedirs(OUTPUT_FOLDER_NAME)
                    print(f"Created folder: {OUTPUT_FOLDER_NAME}")
                except Exception as e:
                    print(f"Error creating folder {OUTPUT_FOLDER_NAME}: {e}")
                    return 

            try:
                df_final_output.to_csv(OUTPUT_CSV_FULL_PATH, index=False)
                print(f"\nSuccessfully saved {len(df_final_output)} filtered symbols and descriptions to {OUTPUT_CSV_FULL_PATH}")
            except Exception as e:
                print(f"Error saving symbols to CSV {OUTPUT_CSV_FULL_PATH}: {e}")
        else:
            print("No symbols matched all filtering criteria. CSV file not created.")
            
    else:
        print(f"No symbols were fetched for keyword '{SYMBOL_API_Q_KEYWORD}'. CSV file not created.")

if __name__ == "__main__":
    main()
